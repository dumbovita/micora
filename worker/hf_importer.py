"""Hugging Face Voice Dataset Importer for Micora.

Allows pulling voice recordings or datasets directly from Hugging Face:
- Supports Model and Dataset repositories (by URL or repo ID).
- Handles ZIP archives containing WAV files (e.g. ailabturkiye/cemyilmaz).
- Handles Parquet-based Hugging Face audio datasets (e.g. omersaidd/tts_mazlum_kiper_tur).
- Downloads only the necessary audio/shards into `datasets/<name>/` (excluded from git).
- Analyzes candidate speech segments using `worker.analyzer.DatasetScanner`.
- Returns structured candidate audio clips for instant voice profile creation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from worker.analyzer import DatasetScanner

logger = logging.getLogger("micora.hf_importer")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aiff"}


def parse_repo_identifier(url_or_id: str) -> tuple[str, str]:
    """Extracts (repo_id, repo_type) from a full URL or repo identifier."""
    raw = url_or_id.strip()
    if raw.startswith("datasets/"):
        return raw[len("datasets/"):], "dataset"

    # Strip URL prefix
    m = re.search(r"huggingface\.co/(datasets/)?([^/]+/[^/?#]+)", raw)
    if m:
        is_dataset = bool(m.group(1))
        repo_id = m.group(2)
        repo_type = "dataset" if is_dataset else "model"
        return repo_id, repo_type

    # Plain repo_id format: username/reponame
    parts = raw.split("/")
    if len(parts) == 2:
        return raw, "model"  # default to probe model first
    elif len(parts) == 3 and parts[0] == "datasets":
        return f"{parts[1]}/{parts[2]}", "dataset"

    return raw, "model"


class HuggingFaceVoiceImporter:
    """Downloads and extracts voice audio from a Hugging Face repository."""

    def __init__(self, output_root: str | Path = "datasets") -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()

    def import_voice(
        self,
        url_or_id: str,
        max_samples: int = 30,
        max_candidates: int = 5,
    ) -> dict[str, Any]:
        raw = url_or_id.strip()

        # Handle YouTube or direct media URLs
        if "youtube.com" in raw or "youtu.be" in raw or (raw.startswith("http") and "huggingface.co" not in raw):
            return self._import_from_url(raw, max_candidates=max_candidates)

        repo_id, candidate_type = parse_repo_identifier(url_or_id)
        logger.info("Resolving Hugging Face repo: %s (type=%s)", repo_id, candidate_type)

        # Detect files in repository
        files = []
        actual_type = candidate_type
        try:
            files = self.api.list_repo_files(repo_id=repo_id, repo_type=actual_type)
        except Exception:
            # Try alternate type
            alternate_type = "dataset" if actual_type == "model" else "model"
            try:
                files = self.api.list_repo_files(repo_id=repo_id, repo_type=alternate_type)
                actual_type = alternate_type
            except Exception as e:
                raise ValueError(f"Could not locate Hugging Face repo '{repo_id}': {e}")

        # If repo had no audio, zip, or parquet files, check alternate type
        has_audio_sources = any(
            f.lower().endswith(tuple(AUDIO_EXTENSIONS)) or f.lower().endswith(".zip") or f.lower().endswith(".parquet")
            for f in files
        )
        if not has_audio_sources and actual_type == "model":
            try:
                ds_files = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                if ds_files:
                    files = ds_files
                    actual_type = "dataset"
            except Exception:
                pass

        logger.info("Found %d files in %s/%s", len(files), actual_type, repo_id)

        # Destination directory for this voice
        safe_name = repo_id.split("/")[-1].replace("-", "_").lower()
        target_dir = self.output_root / safe_name
        target_dir.mkdir(parents=True, exist_ok=True)

        extracted_count = 0

        # Pattern 1: ZIP files with audio inside (e.g. cem.zip in ailabturkiye/cemyilmaz)
        zip_files = [f for f in files if f.lower().endswith(".zip")]
        if zip_files:
            # Pick smallest or most relevant zip
            chosen_zip = zip_files[0]
            logger.info("Downloading zip archive: %s...", chosen_zip)
            local_zip = hf_hub_download(repo_id=repo_id, repo_type=actual_type, filename=chosen_zip)
            has_rvc = False
            with zipfile.ZipFile(local_zip, "r") as z:
                has_rvc = any(m.lower().endswith((".pth", ".index")) for m in z.namelist())
                for member in z.namelist():
                    if any(member.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                        z.extract(member, target_dir)
                        extracted_count += 1
                        if extracted_count >= max_samples:
                            break
            if extracted_count == 0 and has_rvc:
                raise ValueError(
                    f"'{repo_id}' is an RVC voice changer model (.pth), not an audio recording. "
                    "Micora clones voices from speech recordings. "
                    "Paste a YouTube video link of this person speaking into the box instead!"
                )
            logger.info("Extracted %d audio clips from %s", extracted_count, chosen_zip)

        # Pattern 2: Direct audio files
        audio_files = [f for f in files if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)]
        if not extracted_count and audio_files:
            logger.info("Downloading %d direct audio files...", min(len(audio_files), max_samples))
            for af in audio_files[:max_samples]:
                local_path = hf_hub_download(repo_id=repo_id, repo_type=actual_type, filename=af)
                dest_file = target_dir / Path(af).name
                dest_file.write_bytes(Path(local_path).read_bytes())
                extracted_count += 1

        # Pattern 3: Parquet audio dataset (e.g. train-00000-of-*.parquet)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not extracted_count and parquet_files:
            chosen_parquet = parquet_files[0]
            logger.info("Downloading first parquet shard: %s...", chosen_parquet)
            local_pq = hf_hub_download(repo_id=repo_id, repo_type=actual_type, filename=chosen_parquet)

            import pyarrow.parquet as pq
            table = pq.read_table(local_pq)
            col_names = table.column_names

            # Find audio column
            audio_col_name = None
            for c in ["audio", "audio_data", "speech", "wav"]:
                if c in col_names:
                    audio_col_name = c
                    break

            if audio_col_name:
                audio_col = table.column(audio_col_name)
                for i in range(min(table.num_rows, max_samples)):
                    val = audio_col[i].as_py()
                    audio_bytes = None
                    if isinstance(val, dict):
                        audio_bytes = val.get("bytes")
                    elif isinstance(val, (bytes, bytearray)):
                        audio_bytes = bytes(val)

                    if audio_bytes:
                        out_name = f"sample_{i:04d}.wav"
                        out_path = target_dir / out_name
                        out_path.write_bytes(audio_bytes)
                        extracted_count += 1
                logger.info("Extracted %d audio samples from parquet shard", extracted_count)

        if extracted_count == 0:
            raise RuntimeError(f"No audio files could be extracted from {repo_id}")

        # Run DatasetScanner to rank candidates
        scanner = DatasetScanner(target_dir)
        scan_result = scanner.scan(max_candidates=max_candidates)

        # Suggested human-readable name
        voice_title = safe_name.replace("_", " ").title()

        return {
            "type": "hf_imported",
            "repo_id": repo_id,
            "repo_type": actual_type,
            "voice_name": voice_title,
            "folder_path": str(target_dir),
            "extracted_count": extracted_count,
            "scan": scan_result,
        }

    def _import_from_url(self, url: str, max_candidates: int = 5) -> dict[str, Any]:
        """Downloads audio from a YouTube or direct web URL, converts via afconvert, and extracts speech candidates."""
        import subprocess
        import time
        import yt_dlp
        from urllib.parse import parse_qs, urlparse

        # Determine a sensible initial slug from URL or YouTube video ID
        parsed = urlparse(url)
        slug = ""
        if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                slug = f"yt_{qs['v'][0]}"
            elif parsed.path.strip("/"):
                slug = f"yt_{parsed.path.strip('/').split('/')[-1]}"
        if not slug:
            clean_url_slug = re.sub(r"[^a-zA-Z0-9_]", "_", url.split("/")[-1].split("?")[0])[:25]
            slug = clean_url_slug if len(clean_url_slug) >= 3 else f"web_{int(time.time())}"

        voice_title = slug.replace("_", " ").title()

        # Try to extract video title and refine slug before downloading
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if isinstance(info, dict) and info.get("title"):
                    voice_title = str(info["title"]).strip()
                    title_slug = re.sub(r"[^a-zA-Z0-9_]", "_", voice_title.lower()).strip("_")
                    title_slug = re.sub(r"_+", "_", title_slug)[:32]
                    if len(title_slug) >= 3:
                        slug = title_slug
        except Exception as ex:
            logger.warning("Could not query media info prior to download: %s", ex)

        target_dir = self.output_root / slug
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading audio from media URL: %s into %s", url, target_dir)

        ydl_opts = {
            "format": "bestaudio[ext=m4a]/best[ext=mp4]/bestaudio/best",
            "outtmpl": str(target_dir / "raw.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if isinstance(info, dict) and info.get("title"):
                    voice_title = str(info["title"]).strip()
        except Exception as ex:
            raise ValueError(f"Failed to download audio from URL: {ex}")

        raw_files = [p for p in target_dir.glob("raw.*") if p.is_file()]
        if not raw_files:
            raise ValueError("No audio stream could be extracted from URL.")

        raw_file = raw_files[0]
        wav_path = target_dir / "source.wav"
        subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@48000", str(raw_file), str(wav_path)],
            check=True,
            capture_output=True,
        )
        try:
            raw_file.unlink()
        except Exception:
            pass

        scanner = DatasetScanner(target_dir)
        scan_result = scanner.scan(max_candidates=max_candidates)

        return {
            "type": "hf_imported",
            "repo_id": url,
            "repo_type": "url",
            "voice_name": voice_title,
            "folder_path": str(target_dir),
            "extracted_count": 1,
            "scan": scan_result,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face Voice Dataset Importer")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID or URL")
    parser.add_argument("--max-samples", type=int, default=30, help="Max audio files to extract")
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates to return")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    importer = HuggingFaceVoiceImporter()
    result = importer.import_voice(args.repo, max_samples=args.max_samples, max_candidates=args.top)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nImported voice from: {result['repo_id']}")
        print(f"Name: {result['voice_name']}")
        print(f"Extracted: {result['extracted_count']} audio files to {result['folder_path']}")
        print("\n--- Top Reference Candidates ---")
        for idx, c in enumerate(result["scan"]["candidates"], 1):
            print(
                f"{idx}. {c['filename']} ({c['duration_sec']}s, Speech: {int(c['speech_ratio'] * 100)}%, "
                f"Peak: {c['peak_db']} dBFS, Score: {c['score']})"
            )


if __name__ == "__main__":
    main()
