"""Tests for Micora Semantic Text Normalizer.

Covers natural conversational normalization for Turkish (primary) and English (secondary):
- Cardinals & quantities
- Currencies & prices (major, minor, zero-major, thousands)
- Phone numbers (mobile, international, domestic, corporate toll-free)
- Identifiers / verification codes / OTP / PIN / Serials / Tracking IDs (digit-by-digit)
- Explicitly separated digits
- Version numbers and IP addresses
- Units and measurements (storage, weight, distance, temperature, time)
- Natural fractional expressions (1,5 kg -> bir buçuk kilogram, 2,5 saat -> iki buçuk saat, 0,5 litre -> yarım litre)
- Duration quarter expressions (1,25 saat -> bir saat on beş dakika, 1,75 saat -> bir saat kırk beş dakika)
- Fractional counterexamples preserving exact decimal reading (1,3 kg, %1,5, v1.5, 3,14)
- Decimals vs thousands separators
- Dates and digital times
- English conversational quantities, units, fractions, and currencies
- 100% Idempotency guarantee across all test cases
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.normalizer import (
    int_to_english,
    int_to_turkish,
    normalize_english_numbers,
    normalize_text_for_tts,
    normalize_turkish_numbers,
)


def test_turkish_number_primitives():
    assert int_to_turkish(0) == "sıfır"
    assert int_to_turkish(1) == "bir"
    assert int_to_turkish(10) == "on"
    assert int_to_turkish(30) == "otuz"
    assert int_to_turkish(31) == "otuz bir"
    assert int_to_turkish(100) == "yüz"
    assert int_to_turkish(1000) == "bin"
    assert int_to_turkish(1000000) == "bir milyon"
    assert int_to_turkish(2026) == "iki bin yirmi altı"
    assert int_to_turkish(-5) == "eksi beş"


def test_english_number_primitives():
    assert int_to_english(0) == "zero"
    assert int_to_english(30) == "thirty"
    assert int_to_english(42) == "forty two"
    assert int_to_english(100) == "one hundred"
    assert int_to_english(2026) == "two thousand twenty six"
    assert int_to_english(-5) == "minus five"


def test_cardinals_and_quantities():
    cases = [
        ("30", "otuz"),
        ("31", "otuz bir"),
        ("123 elma", "yüz yirmi üç elma"),
        ("250 kişi", "iki yüz elli kişi"),
        ("1000 asker", "bin asker"),
        ("31 çekicem", "otuz bir çekicem"),
        ("deneme deneme 31", "deneme deneme otuz bir"),
        ("otuz bir altmış dokuz seks severim", "otuz bir altmış dokuz seks severim"),
        ("0 ve 1", "sıfır ve bir"),
        ("30'da buluşalım", "otuz'da buluşalım"),
        ("30’a kadar saydım.", "otuz’a kadar saydım."),
        ("-5 derece", "eksi beş derece"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_currencies_and_prices():
    cases = [
        ("250 TL", "iki yüz elli lira"),
        ("250,50 TL", "iki yüz elli lira elli kuruş"),
        ("0,50 TL", "elli kuruş"),
        ("1,05 TL", "bir lira beş kuruş"),
        ("100.000 TL nakit ödeme yapıldı.", "yüz bin lira nakit ödeme yapıldı."),
        ("Toplam borç 1.000.000 TL.", "Toplam borç bir milyon lira."),
        ("Fiyatı 250,50 TL olmuş, yani yaklaşık %20 zam gelmiş.", "Fiyatı iki yüz elli lira elli kuruş olmuş, yani yaklaşık yüzde yirmi zam gelmiş."),
        ("$50", "elli dolar"),
        ("50 $", "elli dolar"),
        ("100 €", "yüz euro"),
        ("€100", "yüz euro"),
        ("£50", "elli sterlin"),
        ("50 £", "elli sterlin"),
        ("250,50 ₺", "iki yüz elli lira elli kuruş"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_phone_numbers():
    cases = [
        ("0555 123 45 67", "sıfır beş yüz elli beş yüz yirmi üç kırk beş altmış yedi"),
        ("+90 532 111 22 33", "artı doksan beş yüz otuz iki yüz on bir yirmi iki otuz üç"),
        ("(0212) 123 45 67", "sıfır iki yüz on iki yüz yirmi üç kırk beş altmış yedi"),
        ("444 0 444", "dört yüz kırk dört sıfır dört yüz kırk dört"),
        ("444 04 44", "dört yüz kırk dört sıfır dört kırk dört"),
        ("0850 222 0 222", "sıfır sekiz yüz elli iki yüz yirmi iki sıfır iki yüz yirmi iki"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_verification_codes_and_otp():
    cases = [
        ("kod: 582910", "kod: beş sekiz iki dokuz bir sıfır"),
        ("onay kodu 123456", "onay kodu bir iki üç dört beş altı"),
        ("şifre: 4829", "şifre: dört sekiz iki dokuz"),
        ("güvenlik kodu: 9021", "güvenlik kodu: dokuz sıfır iki bir"),
        ("seri no: 948271", "seri no: dokuz dört sekiz iki yedi bir"),
        ("takip no: 12345678", "takip no: bir iki üç dört beş altı yedi sekiz"),
        ("TC no: 12345678901", "TC no: bir iki üç dört beş altı yedi sekiz dokuz sıfır bir"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_explicit_separated_digits():
    cases = [
        ("1 2 3", "bir iki üç"),
        ("9 8 7 6", "dokuz sekiz yedi altı"),
        ("5 4 3 2 1", "beş dört üç iki bir"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_versions_and_ip_addresses():
    cases = [
        ("v1.2.7", "v bir nokta iki nokta yedi"),
        ("sürüm 2.0", "sürüm iki nokta sıfır"),
        ("192.168.1.1", "yüz doksan iki nokta yüz altmış sekiz nokta bir nokta bir"),
        ("10.0.0.1", "on nokta sıfır nokta sıfır nokta bir"),
        ("0.9.1", "sıfır nokta dokuz nokta bir"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_units_and_measurements():
    cases = [
        ("30 GB", "otuz gigabayt"),
        ("30gb", "otuz gigabayt"),
        ("qemudan dene bi 30gb alan versen yeter", "qemudan dene bi otuz gigabayt alan versen yeter"),
        ("1,5 kg", "bir buçuk kilogram"),
        ("0,5 kg", "yarım kilogram"),
        ("2,5 saat", "iki buçuk saat"),
        ("0,5 saat", "yarım saat"),
        ("1,25 saat", "bir saat on beş dakika"),
        ("1,75 saat", "bir saat kırk beş dakika"),
        ("0,5 litre", "yarım litre"),
        ("1,5 litre", "bir buçuk litre"),
        ("1,5 metre", "bir buçuk metre"),
        ("0,5 metre", "yarım metre"),
        ("3,5 km", "üç buçuk kilometre"),
        ("0,5 porsiyon", "yarım porsiyon"),
        ("1,5 paket", "bir buçuk paket"),
        ("25 °C", "yirmi beş derece"),
        ("-5 °C", "eksi beş derece"),
        ("10 km", "on kilometre"),
        ("5 dk", "beş dakika"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_fractional_counterexamples():
    cases = [
        ("1,3 kg", "bir virgül üç kilogram"),
        ("0,125 litre", "sıfır virgül yüz yirmi beş litre"),
        ("2,7 saat", "iki virgül yedi saat"),
        ("1,5", "bir virgül beş"),
        ("%1,5", "yüzde bir virgül beş"),
        ("v1.5", "v bir nokta beş"),
        ("3,14", "üç virgül on dört"),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_decimals_vs_thousands():
    cases = [
        ("100.000", "yüz bin"),
        ("1.234.567", "bir milyon iki yüz otuz dört bin beş yüz altmış yedi"),
        ("3,14", "üç virgül on dört"),
        ("0,5", "sıfır virgül beş"),
        ("0,05", "sıfır virgül sıfır beş"),
        ("Pi sayısı yaklaşık 3.14 olarak kabul edilir.", "Pi sayısı yaklaşık üç nokta on dört olarak kabul edilir."),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_dates_and_times():
    cases = [
        ("15 Mart 2026 tarihinde saat 14:30'da toplantımız başlayacak.", "on beş Mart iki bin yirmi altı tarihinde saat on dört otuz'da toplantımız başlayacak."),
        ("Saat 09:05 oldu.", "Saat dokuz sıfır beş oldu."),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"


def test_english_conversational():
    cases = [
        ("I have 30 apples in the basket.", "I have thirty apples in the basket."),
        ("The cost is $250.50 today.", "The cost is two hundred fifty dollars and fifty cents today."),
        ("Hello, wait 5 minutes with that voice cloning.", "Hello, wait five minutes with that voice cloning."),
        ("Your verification code is 582910.", "Your verification code is five eight two nine one zero."),
        ("Your tracking number is 123456.", "Your tracking number is one two three four five six."),
        ("Server IP is 192.168.1.1 with 30 GB storage.", "Server IP is one hundred ninety two point one hundred sixty eight point one point one with thirty gigabytes storage."),
        ("We bought 1.5 kg of apples.", "We bought one and a half kilograms of apples."),
        ("We drank 0.5 liter of water.", "We drank half a liter of water."),
        ("The meeting lasted 2.5 hours.", "The meeting lasted two and a half hours."),
        ("Please wait 0.5 hour.", "Please wait half an hour."),
        ("The session took 1.25 hours.", "The session took one hour and fifteen minutes."),
        ("The flight is 1.75 hours long.", "The flight is one hour and forty-five minutes long."),
        ("The package weighs 1.3 kg.", "The package weighs one point three kilograms."),
        ("There is 0.8 liter of water left.", "There is zero point eight liters of water left."),
        ("The interest rate is 1.5% today.", "The interest rate is one point five percent today."),
        ("Please download v1.5 now.", "Please download v one point five now."),
    ]
    for inp, expected in cases:
        out = normalize_text_for_tts(inp)
        assert out == expected, f"Failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_text_for_tts(out) == out, f"Idempotency failed: '{out}' mutated on second pass"

    # Direct unit checks on English normalization engine
    direct_en_cases = [
        ("1.5 kg", "one and a half kilograms"),
        ("0.5 liter", "half a liter"),
        ("2.5 hours", "two and a half hours"),
        ("0.5 hour", "half an hour"),
        ("1.25 hours", "one hour and fifteen minutes"),
        ("1.75 hours", "one hour and forty-five minutes"),
        ("1.3 kg", "one point three kilograms"),
        ("0.8 liter", "zero point eight liters"),
        ("1.5%", "one point five percent"),
        ("v1.5", "v one point five"),
        ("serial no: 948271", "serial no: nine four eight two seven one"),
        ("order id: 84920", "order id: eight four nine two zero"),
    ]
    for inp, expected in direct_en_cases:
        out = normalize_english_numbers(inp)
        assert out == expected, f"Direct EN failed: '{inp}' -> expected '{expected}', got '{out}'"
        assert normalize_english_numbers(out) == out, f"Direct EN idempotency failed: '{out}'"


def test_validation_corpus_samples():
    import json
    corpus_path = Path(__file__).resolve().parent.parent / "corpus" / "validation_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data.get("samples", []):
        text = sample["text"]
        norm1 = normalize_text_for_tts(text)
        norm2 = normalize_text_for_tts(norm1)
        assert norm1 == norm2, f"Corpus sample '{sample['id']}' failed idempotency: '{norm1}' != '{norm2}'"
        print(f"  [Corpus {sample['id']}]: '{text}'\n    -> '{norm1}'")


if __name__ == "__main__":
    test_turkish_number_primitives()
    test_english_number_primitives()
    test_cardinals_and_quantities()
    test_currencies_and_prices()
    test_phone_numbers()
    test_verification_codes_and_otp()
    test_explicit_separated_digits()
    test_versions_and_ip_addresses()
    test_units_and_measurements()
    test_fractional_counterexamples()
    test_decimals_vs_thousands()
    test_dates_and_times()
    test_english_conversational()
    test_validation_corpus_samples()
    print("All python semantic normalization and idempotency tests passed successfully!")
