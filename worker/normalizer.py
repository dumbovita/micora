"""Micora Semantic Text Normalization Module.

Provides context-aware semantic normalization for Turkish (primary) and English (secondary).
Converts digits, percentages, timestamps, dates, currencies, phone numbers, versions,
IP addresses, verification codes, units, and numeric expressions into natural conversational
spoken words before feeding into TTS models (Chatterbox, MOSS-TTS).

Preserves the invariant that semantic normalization is idempotent and occurs exactly once
in the normal synthesis runtime path.
"""

from __future__ import annotations

import re


# =============================================================================
# 1. Base Number-to-Words Primitives
# =============================================================================

def int_to_turkish(n: int) -> str:
    """Convert an integer to natural Turkish spoken cardinal words.

    Examples:
        0 -> 'sıfır'
        1 -> 'bir'
        30 -> 'otuz'
        31 -> 'otuz bir'
        100 -> 'yüz' (never 'bir yüz')
        1000 -> 'bin' (never 'bir bin')
        1000000 -> 'bir milyon'
        2026 -> 'iki bin yirmi altı'
    """
    if n == 0:
        return "sıfır"

    units = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
    tens = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
    scales = [
        "", "bin", "milyon", "milyar", "trilyon",
        "katrilyon", "kentilyon", "sekstilyon", "septilyon"
    ]

    if n < 0:
        return "eksi " + int_to_turkish(abs(n))

    parts = []
    chunk_idx = 0
    temp = n

    while temp > 0:
        chunk = temp % 1000
        temp //= 1000

        if chunk > 0:
            c_parts = []
            h = chunk // 100
            t = (chunk % 100) // 10
            u = chunk % 10

            if h == 1:
                c_parts.append("yüz")
            elif h > 1:
                c_parts.append(units[h] + " yüz")

            if t > 0:
                c_parts.append(tens[t])

            if u > 0:
                # 1000 is 'bin', never 'bir bin'
                if not (chunk_idx == 1 and chunk == 1):
                    c_parts.append(units[u])

            chunk_str = " ".join(c_parts)
            if chunk_idx > 0:
                scale_name = scales[chunk_idx] if chunk_idx < len(scales) else ""
                if scale_name:
                    if chunk_str:
                        chunk_str = f"{chunk_str} {scale_name}"
                    else:
                        chunk_str = scale_name

            parts.insert(0, chunk_str)

        chunk_idx += 1

    return " ".join(parts).strip()


def int_to_english(n: int) -> str:
    """Convert an integer to natural English spoken cardinal words."""
    if n == 0:
        return "zero"

    units = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["", "thousand", "million", "billion", "trillion"]

    if n < 0:
        return "minus " + int_to_english(abs(n))

    if n < 20:
        return units[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else " " + units[n % 10])
    if n < 1000:
        return units[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_to_english(n % 100))

    for i, scale in enumerate(scales[1:], 1):
        bound = 1000 ** (i + 1)
        if n < bound:
            base = 1000 ** i
            rem = n % base
            return int_to_english(n // base) + " " + scale + ("" if rem == 0 else " " + int_to_english(rem))

    return str(n)


def digits_to_turkish(s: str) -> str:
    """Convert a sequence of digits into individual Turkish digit words."""
    mapping = {
        "0": "sıfır", "1": "bir", "2": "iki", "3": "üç", "4": "dört",
        "5": "beş", "6": "altı", "7": "yedi", "8": "sekiz", "9": "dokuz"
    }
    return " ".join(mapping[c] for c in s if c in mapping)


def digits_to_english(s: str) -> str:
    """Convert a sequence of digits into individual English digit words."""
    mapping = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    return " ".join(mapping[c] for c in s if c in mapping)


# =============================================================================
# 2. Language Detection
# =============================================================================

def is_english_text(text: str) -> bool:
    """Heuristic check whether text appears to be primarily English."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    if any(c in turkish_chars for c in text):
        return False
    common_en = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "shall", "should", "may",
        "might", "must", "can", "could", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "up", "about", "into", "over", "after", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they", "my",
        "your", "his", "her", "its", "our", "their", "what", "which", "who", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "hello", "hi", "please", "thanks", "thank", "today",
        "yesterday", "tomorrow", "now", "here", "there", "water", "hour", "hours",
        "minute", "minutes", "second", "seconds", "day", "days", "week", "weeks",
        "month", "months", "year", "years", "time", "times", "liter", "liters",
        "litre", "litres", "kilogram", "kilograms", "meter", "meters", "kilometer",
        "kilometers", "gram", "grams", "ton", "tons", "dollar", "dollars", "cent",
        "cents", "euro", "euros", "pound", "pounds", "percent", "cost", "price",
        "total", "code", "pin", "otp", "serial", "number", "tracking", "order", "id",
        "weighs", "weigh", "package", "left", "remaining", "bought", "drank",
        "lasted", "took", "flight", "exam", "session", "meeting", "apples", "basket",
        "storage", "server", "port", "download", "voice", "cloning", "apple", "silicon", "rate"
    }
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if len(words & common_en) >= 2:
        return True
    return False


# =============================================================================
# 3. Currency Formatters
# =============================================================================

def format_currency_turkish(whole: int, frac: int, frac_len: int, curr_type: str, suffix: str = "") -> str:
    """Format monetary expressions into natural Turkish conversational speech.

    Examples:
        250 TL -> 'iki yüz elli lira'
        250,50 TL -> 'iki yüz elli lira elli kuruş'
        0,50 TL -> 'elli kuruş'
        1,05 TL -> 'bir lira beş kuruş'
        1 TL -> 'bir lira'
    """
    if frac_len == 1:
        frac *= 10

    curr_upper = curr_type.upper()
    if curr_upper in ("TL", "TRY", "₺", "LIRA"):
        major = "lira"
        minor = "kuruş"
    elif curr_upper in ("USD", "$", "DOLAR", "DOLLAR"):
        major = "dolar"
        minor = "sent"
    elif curr_upper in ("EUR", "€", "AVRO", "EURO"):
        major = "euro"
        minor = "sent"
    elif curr_upper in ("GBP", "£", "STERLIN", "POUND"):
        major = "sterlin"
        minor = "peni"
    else:
        major = curr_type.lower()
        minor = "kuruş"

    parts = []
    if whole > 0 or frac == 0:
        parts.append(f"{int_to_turkish(whole)} {major}")
    if frac > 0:
        parts.append(f"{int_to_turkish(frac)} {minor}")

    res = " ".join(parts)
    if suffix:
        res = f"{res}{suffix}"
    return res


def format_currency_english(whole: int, frac: int, frac_len: int, curr_type: str) -> str:
    """Format monetary expressions into natural English conversational speech.

    Examples:
        $250 -> 'two hundred fifty dollars'
        $250.50 -> 'two hundred fifty dollars and fifty cents'
        $0.50 -> 'fifty cents'
        $1 -> 'one dollar'
    """
    if frac_len == 1:
        frac *= 10

    curr_upper = curr_type.upper()
    if curr_upper in ("USD", "$", "DOLLAR", "DOLAR"):
        major_singular = "dollar"
        major_plural = "dollars"
        minor = "cents"
    elif curr_upper in ("EUR", "€", "EURO"):
        major_singular = "euro"
        major_plural = "euros"
        minor = "cents"
    elif curr_upper in ("GBP", "£", "POUND", "STERLIN"):
        major_singular = "pound"
        major_plural = "pounds"
        minor = "pence"
    else:
        major_singular = curr_type.lower()
        major_plural = curr_type.lower() + "s"
        minor = "cents"

    parts = []
    if whole > 0 or frac == 0:
        unit = major_singular if whole == 1 else major_plural
        parts.append(f"{int_to_english(whole)} {unit}")
    if frac > 0:
        parts.append(f"{int_to_english(frac)} {minor}")

    if whole > 0 and frac > 0:
        return f"{parts[0]} and {parts[1]}"
    return " ".join(parts)


# =============================================================================
# 4. Quantity, Measurement & Duration Formatters
# =============================================================================

def format_quantity_with_unit_turkish(raw_val: str, raw_unit: str) -> str:
    """Format physical quantities, durations, and measurements into natural Turkish words.

    Conversational fractional expressions are applied contextually to everyday units:
        1,5 kg -> 'bir buçuk kilogram'
        2,5 saat -> 'iki buçuk saat'
        0,5 litre -> 'yarım litre'
        1,5 metre -> 'bir buçuk metre'
        0,5 saat -> 'yarım saat'
        1,25 saat -> 'bir saat on beş dakika'
        1,75 saat -> 'bir saat kırk beş dakika'

    Exact decimal readings are strictly preserved for non-half/quarter quantities
    and precision-sensitive measurements (never approximated or rounded!):
        1,3 kg -> 'bir virgül üç kilogram'
        0,8 litre -> 'sıfır virgül sekiz litre'
    """
    unit_map_tr = {
        "gb": "gigabayt", "gigabayt": "gigabayt",
        "mb": "megabayt", "megabayt": "megabayt",
        "kb": "kilobayt", "kilobayt": "kilobayt",
        "tb": "terabayt", "terabayt": "terabayt",
        "kg": "kilogram", "kilogram": "kilogram", "kilo": "kilo",
        "mg": "miligram", "miligram": "miligram",
        "gram": "gram", "gr": "gram",
        "ton": "ton",
        "km": "kilometre", "kilometre": "kilometre",
        "m": "metre", "metre": "metre",
        "cm": "santimetre", "santimetre": "santimetre",
        "mm": "milimetre", "milimetre": "milimetre",
        "litre": "litre", "lt": "litre", "l": "litre",
        "saat": "saat", "sa": "saat",
        "dakika": "dakika", "dk": "dakika",
        "saniye": "saniye", "sn": "saniye",
        "gün": "gün", "hafta": "hafta", "ay": "ay", "yıl": "yıl", "sene": "sene",
        "derece": "derece", "°c": "derece", "° c": "derece",
        "porsiyon": "porsiyon", "bardak": "bardak", "kaşık": "kaşık", "dilim": "dilim",
        "paket": "paket", "kutu": "kutu", "şişe": "şişe",
    }
    unit_norm = raw_unit.lower().replace(" ", "")
    unit_name = unit_map_tr.get(unit_norm, unit_norm)

    is_neg = raw_val.startswith("-")
    clean_val = raw_val.lstrip("-")

    if "," in clean_val or "." in clean_val:
        sep = "," if "," in clean_val else "."
        w_str, f_str = clean_val.split(sep, 1)
        w = int(w_str)

        # 1. Exact half fractions: 0,5 / 1,5 / 2,5 ...
        if f_str in ("5", "50", "500"):
            if w == 0:
                words = f"yarım {unit_name}"
            else:
                words = f"{int_to_turkish(w)} buçuk {unit_name}"
        # 2. Exact quarter fractions for durations: 0,25 / 1,25 / 1,75 saat
        elif unit_name == "saat" and f_str in ("25", "250"):
            if w == 0:
                words = "on beş dakika"
            elif w == 1:
                words = "bir saat on beş dakika"
            else:
                words = f"{int_to_turkish(w)} saat on beş dakika"
        elif unit_name == "saat" and f_str in ("75", "750"):
            if w == 0:
                words = "kırk beş dakika"
            elif w == 1:
                words = "bir saat kırk beş dakika"
            else:
                words = f"{int_to_turkish(w)} saat kırk beş dakika"
        # 3. All other exact decimals (never approximate or round)
        else:
            leading_zeros = len(f_str) - len(f_str.lstrip("0"))
            if leading_zeros > 0:
                zero_words = " ".join(["sıfır"] * leading_zeros)
                f_val = f_str.lstrip("0")
                f_words = f"{zero_words} {int_to_turkish(int(f_val))}" if f_val else zero_words
            else:
                f_words = int_to_turkish(int(f_str))
            words = f"{int_to_turkish(w)} virgül {f_words} {unit_name}"
    else:
        w = int(clean_val)
        words = f"{int_to_turkish(w)} {unit_name}"

    if is_neg:
        return f"eksi {words}"
    return words


def format_quantity_with_unit_english(raw_val: str, raw_unit: str) -> str:
    """Format physical quantities, durations, and measurements into natural English words.

    Conversational fractional expressions are applied contextually to everyday units:
        1.5 kg -> 'one and a half kilograms'
        0.5 liter -> 'half a liter'
        2.5 hours -> 'two and a half hours'
        0.5 hour -> 'half an hour'
        1.25 hours -> 'one hour and fifteen minutes'
        1.75 hours -> 'one hour and forty-five minutes'

    Exact decimal readings are strictly preserved for non-half/quarter quantities
    and precision-sensitive measurements (never approximated or rounded!):
        1.3 kg -> 'one point three kilograms'
        0.8 liter -> 'zero point eight liters'
    """
    unit_map_en = {
        "gb": ("gigabyte", "gigabytes"),
        "mb": ("megabyte", "megabytes"),
        "kb": ("kilobyte", "kilobytes"),
        "tb": ("terabyte", "terabytes"),
        "kg": ("kilogram", "kilograms"),
        "kilogram": ("kilogram", "kilograms"),
        "kilograms": ("kilogram", "kilograms"),
        "g": ("gram", "grams"),
        "gram": ("gram", "grams"),
        "grams": ("gram", "grams"),
        "mg": ("milligram", "milligrams"),
        "ton": ("ton", "tons"),
        "tons": ("ton", "tons"),
        "km": ("kilometer", "kilometers"),
        "kilometer": ("kilometer", "kilometers"),
        "kilometers": ("kilometer", "kilometers"),
        "m": ("meter", "meters"),
        "meter": ("meter", "meters"),
        "meters": ("meter", "meters"),
        "cm": ("centimeter", "centimeters"),
        "mm": ("millimeter", "millimeters"),
        "liter": ("liter", "liters"),
        "liters": ("liter", "liters"),
        "litre": ("litre", "litres"),
        "litres": ("litre", "litres"),
        "l": ("liter", "liters"),
        "hour": ("hour", "hours"),
        "hours": ("hour", "hours"),
        "hr": ("hour", "hours"),
        "hrs": ("hour", "hours"),
        "minute": ("minute", "minutes"),
        "minutes": ("minute", "minutes"),
        "min": ("minute", "minutes"),
        "second": ("second", "seconds"),
        "seconds": ("second", "seconds"),
        "sec": ("second", "seconds"),
        "day": ("day", "days"),
        "days": ("day", "days"),
        "week": ("week", "weeks"),
        "weeks": ("week", "weeks"),
        "month": ("month", "months"),
        "months": ("month", "months"),
        "year": ("year", "years"),
        "years": ("year", "years"),
        "°c": ("degree", "degrees"),
        "° c": ("degree", "degrees"),
        "degree": ("degree", "degrees"),
        "degrees": ("degree", "degrees"),
    }
    unit_norm = raw_unit.lower().replace(" ", "")
    names = unit_map_en.get(unit_norm)
    if not names:
        return f"{raw_val} {raw_unit}"

    u_sing, u_plur = names

    is_neg = raw_val.startswith("-")
    clean_val = raw_val.lstrip("-")

    if "." in clean_val:
        w_str, f_str = clean_val.split(".", 1)
        w = int(w_str)

        # 1. Exact half fractions: 0.5 / 1.5 / 2.5 ...
        if f_str in ("5", "50", "500"):
            if w == 0:
                if u_sing in ("liter", "litre"):
                    words = "half a liter"
                elif u_sing == "hour":
                    words = "half an hour"
                else:
                    words = f"half a {u_sing}"
            elif w == 1:
                words = f"one and a half {u_plur}"
            else:
                words = f"{int_to_english(w)} and a half {u_plur}"
        # 2. Exact quarter fractions for durations: 0.25 / 1.25 / 1.75 hours
        elif u_sing == "hour" and f_str in ("25", "250"):
            if w == 0:
                words = "fifteen minutes"
            elif w == 1:
                words = "one hour and fifteen minutes"
            else:
                words = f"{int_to_english(w)} hours and fifteen minutes"
        elif u_sing == "hour" and f_str in ("75", "750"):
            if w == 0:
                words = "forty-five minutes"
            elif w == 1:
                words = "one hour and forty-five minutes"
            else:
                words = f"{int_to_english(w)} hours and forty-five minutes"
        # 3. All other exact decimals (never approximate or round)
        else:
            f_words = " ".join(digits_to_english(d) for d in f_str)
            words = f"{int_to_english(w)} point {f_words} {u_plur}"
    else:
        w = int(clean_val)
        unit_word = u_sing if w == 1 else u_plur
        words = f"{int_to_english(w)} {unit_word}"

    if is_neg:
        return f"minus {words}"
    return words


# =============================================================================
# 5. Turkish Semantic Normalization
# =============================================================================

def normalize_turkish_numbers(text: str) -> str:
    """Normalize numeric expressions in Turkish text into natural conversational words.

    Follows strict precedence rules to prevent broad patterns from corrupting structured tokens:
    1. Protect URLs, emails, and social handles
    2. IP addresses (192.168.1.1)
    3. Prefixed versions (v1.2.7, sürüm 2.0)
    4. Verification codes / OTP / PIN (kod: 582910, şifre: 4829)
    5. Telephone numbers (mobile, international, corporate)
    6. Currency expressions (250 TL, 250,50 TL, 0,50 TL, $50, 100 €)
    7. Percentages (%20, %20,5)
    8. Turkish thousands separators (100.000, 1.000.000, 1.234.567)
    9. Standalone version numbers (1.2.7, 0.9.1)
    10. Units & measurements (30 GB, 1,5 kg, 25 °C, 10 km, 5 dk)
    11. Explicitly separated digits (1 2 3, 9 8 7 6)
    12. Digital times (14:30, 14:30'da)
    13. Dates (15 Mart 2026)
    14. Decimal commas and dots (3,14, 0,5)
    15. Suffixes attached to numbers (30'da, 30’a)
    16. Negative integers (-5)
    17. Standalone cardinal integers (30, 250, 1000)
    18. Restore protected URLs, emails, and handles
    """
    # 1. Protection of URLs, emails, and handles
    placeholders: dict[str, str] = {}
    p_counter = 0

    def protect_match(m: re.Match) -> str:
        nonlocal p_counter
        ph = f"__MICORA_PROT_{p_counter}__"
        placeholders[ph] = m.group(0)
        p_counter += 1
        return ph

    text = re.sub(r"https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|io|dev|edu|gov)(?:/\S*)?", protect_match, text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", protect_match, text)
    text = re.sub(r"@[a-zA-Z0-9_]+", protect_match, text)

    # 2. IP addresses (4 octets 0-255)
    def repl_ip(m: re.Match) -> str:
        parts = m.group(0).split(".")
        return " nokta ".join(int_to_turkish(int(p)) for p in parts)

    text = re.sub(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b", repl_ip, text)

    # 3. Prefixed version numbers: v1.2.7, sürüm 2.0, versiyon 3.1.2
    def repl_version_prefixed(m: re.Match) -> str:
        prefix = m.group(1)
        ver_nums = m.group(2).split(".")
        ver_words = " nokta ".join(int_to_turkish(int(v)) for v in ver_nums)
        return f"{prefix} {ver_words}"

    text = re.sub(r"(?i)\b(v|ver|version|versiyon|sürüm)\s*(\d+(?:\.\d+)+)\b", repl_version_prefixed, text)

    # 4. Identifiers, verification codes, OTP, PIN, serials: kod: 582910, şifre: 4829, onay kodu 123456, seri no: 948271
    def repl_otp(m: re.Match) -> str:
        kw = m.group(1)
        sep = m.group(2)
        digits = m.group(3).replace(" ", "")
        return f"{kw}{sep}{digits_to_turkish(digits)}"

    text = re.sub(
        r"(?i)\b(kod(?:u|unuz)?|şifre(?:si|niz)?|pin(?:\s*kodu(?:nuz)?)?|onay\s*kodu(?:nuz)?|doğrulama\s*kodu(?:nuz)?|güvenlik\s*kodu(?:nuz)?|sms\s*kodu(?:nuz)?|otp|code|verification\s*code|seri\s*no(?:su)?|seri\s*numarası|serial\s*no|kullanıcı\s*id|müşteri\s*no|hesap\s*no|takip\s*no|kargo\s*takip\s*no|sipariş\s*no|tc\s*no|tc\s*kimlik(?:\s*no)?|kimlik\s*no|barkod|imei(?:\s*no)?)([:\s]+)(\d{3,16}|\d{3,4}(?:\s+\d{3,4})+)\b",
        repl_otp,
        text,
    )

    # 5. Telephone numbers:
    def phone_block_tr(block: str) -> str:
        """Format a phone number digit block into Turkish words, preserving leading zeros."""
        if not block:
            return ""
        if block.startswith("0") and len(block) > 1:
            zeros = len(block) - len(block.lstrip("0"))
            rest = block.lstrip("0")
            zero_words = " ".join(["sıfır"] * zeros)
            if rest:
                return f"{zero_words} {int_to_turkish(int(rest))}"
            return zero_words
        return int_to_turkish(int(block))

    # 5a. International: +90 555 123 45 67, +90 532 111 22 33
    def repl_phone_int(m: re.Match) -> str:
        cc, area, g1, g2, g3 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        prefix_words = "artı doksan" if cc in ("90", "+90", "0090") else f"artı {phone_block_tr(cc.replace('+', ''))}"
        return f"{prefix_words} {phone_block_tr(area)} {phone_block_tr(g1)} {phone_block_tr(g2)} {phone_block_tr(g3)}"

    text = re.sub(r"(?:\+|00)(90)\s*\(?([2-589]\d{2})\)?[\s.-]?(\d{3})[\s.-]?(\d{2})[\s.-]?(\d{2})\b", repl_phone_int, text)

    # 5b. Domestic standard: 0555 123 45 67, 0 (555) 123 45 67, (0212) 123 45 67, 05551234567
    def repl_phone_dom(m: re.Match) -> str:
        area, g1, g2, g3 = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"sıfır {phone_block_tr(area)} {phone_block_tr(g1)} {phone_block_tr(g2)} {phone_block_tr(g3)}"

    text = re.sub(r"(?:^|(?<=\s))0\s*\(?([2-589]\d{2})\)?[\s.-]?(\d{3})[\s.-]?(\d{2})[\s.-]?(\d{2})\b", repl_phone_dom, text)
    text = re.sub(r"\(\s*0\s*([2-589]\d{2})\s*\)[\s.-]?(\d{3})[\s.-]?(\d{2})[\s.-]?(\d{2})\b", repl_phone_dom, text)

    # 5c. Corporate toll-free: 444 0 444, 444 04 44, 0850 222 0 222
    def repl_phone_corp(m: re.Match) -> str:
        p1, p2, p3 = m.group(1), m.group(2), m.group(3)
        prefix = phone_block_tr(p1)
        return f"{prefix} {phone_block_tr(p2)} {phone_block_tr(p3)}"

    text = re.sub(r"\b(444|0850)[\s.-](\d{1,4})[\s.-](\d{1,4})\b", repl_phone_corp, text)

    # 6. Currency expressions:
    def repl_curr_post(m: re.Match) -> str:
        raw_num = m.group(1)
        curr = m.group(2)
        apostrophe = m.group(3) or ""
        suffix = m.group(4) or ""

        if re.match(r"^\d{1,3}(?:\.\d{3})+$", raw_num):
            whole = int(raw_num.replace(".", ""))
            frac = 0
            frac_len = 0
        elif "," in raw_num:
            parts = raw_num.split(",")
            whole = int(parts[0])
            frac = int(parts[1])
            frac_len = len(parts[1])
        elif "." in raw_num and len(raw_num.split(".")[1]) <= 2:
            parts = raw_num.split(".")
            whole = int(parts[0])
            frac = int(parts[1])
            frac_len = len(parts[1])
        else:
            whole = int(raw_num)
            frac = 0
            frac_len = 0

        sfx = f"{apostrophe}{suffix}" if suffix else ""
        return format_currency_turkish(whole, frac, frac_len, curr, sfx)

    text = re.sub(r"\b(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)\s*(TL|tl|₺|TRY|lira|USD|usd|\$|EUR|eur|€|GBP|gbp|£)(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?(?=$|[^\w])", repl_curr_post, text)

    def repl_curr_pre(m: re.Match) -> str:
        curr = m.group(1)
        raw_num = m.group(2)
        apostrophe = m.group(3) or ""
        suffix = m.group(4) or ""

        if re.match(r"^\d{1,3}(?:\.\d{3})+$", raw_num):
            whole = int(raw_num.replace(".", ""))
            frac = 0
            frac_len = 0
        elif "," in raw_num:
            parts = raw_num.split(",")
            whole = int(parts[0])
            frac = int(parts[1])
            frac_len = len(parts[1])
        elif "." in raw_num and len(raw_num.split(".")[1]) <= 2:
            parts = raw_num.split(".")
            whole = int(parts[0])
            frac = int(parts[1])
            frac_len = len(parts[1])
        else:
            whole = int(raw_num)
            frac = 0
            frac_len = 0

        sfx = f"{apostrophe}{suffix}" if suffix else ""
        return format_currency_turkish(whole, frac, frac_len, curr, sfx)

    text = re.sub(r"(₺|\$|€|£)\s*(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?(?=$|[^\w])", repl_curr_pre, text)

    # 7. Percentages: %20, % 30, %20,5, 20%
    def repl_percent_prefix(m: re.Match) -> str:
        raw_num = m.group(1)
        apostrophe = m.group(2) or ""
        suffix = m.group(3) or ""
        if "," in raw_num:
            w, f = raw_num.split(",")
            val_str = f"{int_to_turkish(int(w))} virgül {int_to_turkish(int(f))}"
        elif "." in raw_num:
            w, f = raw_num.split(".")
            val_str = f"{int_to_turkish(int(w))} nokta {int_to_turkish(int(f))}"
        else:
            val_str = int_to_turkish(int(raw_num))
        sfx = f"{apostrophe}{suffix}" if suffix else ""
        return f"yüzde {val_str}{sfx}"

    text = re.sub(r"%\s*(\d+(?:[.,]\d+)?)(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?\b", repl_percent_prefix, text)

    def repl_percent_postfix(m: re.Match) -> str:
        raw_num = m.group(1)
        apostrophe = m.group(2) or ""
        suffix = m.group(3) or ""
        if "," in raw_num:
            w, f = raw_num.split(",")
            val_str = f"{int_to_turkish(int(w))} virgül {int_to_turkish(int(f))}"
        elif "." in raw_num:
            w, f = raw_num.split(".")
            val_str = f"{int_to_turkish(int(w))} nokta {int_to_turkish(int(f))}"
        else:
            val_str = int_to_turkish(int(raw_num))
        sfx = f"{apostrophe}{suffix}" if suffix else ""
        return f"yüzde {val_str}{sfx}"

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?\b", repl_percent_postfix, text)

    # 8. Turkish thousands separators: 100.000, 1.000.000, 1.234.567
    def repl_turkish_thousands(m: re.Match) -> str:
        raw_num = m.group(0).replace(".", "")
        return int_to_turkish(int(raw_num))

    text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", repl_turkish_thousands, text)

    # 9. Standalone version numbers (not thousands numbers): 1.2.7, 0.9.1
    def repl_version_standalone(m: re.Match) -> str:
        ver_nums = m.group(1).split(".")
        return " nokta ".join(int_to_turkish(int(v)) for v in ver_nums)

    text = re.sub(r"\b(\d+\.\d+\.\d+(?:\.\d+)*)\b", repl_version_standalone, text)

    # 10. Units, physical quantities, durations, and measurements
    def repl_units(m: re.Match) -> str:
        return format_quantity_with_unit_turkish(m.group(1), m.group(2))

    text = re.sub(
        r"(-?\d+(?:[.,]\d+)?)\s*(GB|gb|gB|Gb|MB|mb|KB|kb|TB|tb|kg|kilogram|kilo|ton|mg|km|kilometre|metre|cm|santimetre|mm|milimetre|litre|lt|°C|°\s*C|derece|saat|sa|dakika|dk|saniye|sn|gün|hafta|ay|yıl|sene|porsiyon|bardak|kaşık|dilim|paket|kutu|şişe)\b",
        repl_units,
        text,
    )
    text = re.sub(r"(-?\d+(?:[.,]\d+)?)\s+([ml])\b", repl_units, text)

    # 11. Explicit separated digits: 1 2 3, 9 8 7 6, 1-2-3
    def repl_sep_digits(m: re.Match) -> str:
        raw = m.group(0)
        digits = [c for c in raw if c.isdigit()]
        return " ".join(digits_to_turkish(d) for d in digits)

    text = re.sub(r"\b\d(?:\s+\d)+\b", repl_sep_digits, text)
    text = re.sub(r"\b\d(?:-\d){2,}\b", repl_sep_digits, text)

    # 12. Time expressions: 14:30, 14:30'da, 09:15, 08:00'de
    def repl_time(m: re.Match) -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        apostrophe = m.group(3) or ""
        suffix = m.group(4) or ""
        if mn == 0:
            time_words = int_to_turkish(h)
        elif mn < 10:
            time_words = f"{int_to_turkish(h)} sıfır {int_to_turkish(mn)}"
        else:
            time_words = f"{int_to_turkish(h)} {int_to_turkish(mn)}"

        if suffix:
            return f"{time_words}{apostrophe}{suffix}"
        return time_words

    text = re.sub(r"\b(\d{1,2}):(\d{2})(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?\b", repl_time, text)

    # 13. Turkish Dates: 15 Mart 2026, 15 Mart 2026'da
    def repl_date(m: re.Match) -> str:
        day = int(m.group(1))
        month = m.group(2)
        year = int(m.group(3))
        apostrophe = m.group(4) or ""
        suffix = m.group(5) or ""
        res = f"{int_to_turkish(day)} {month} {int_to_turkish(year)}"
        if suffix:
            res = f"{res}{apostrophe}{suffix}"
        return res

    text = re.sub(r"\b(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\d{4})(?:(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+))?\b", repl_date, text)

    # 14. Decimal commas (non-currency): 3,14 -> üç virgül on dört, 0,5 -> sıfır virgül beş
    def repl_decimal_comma(m: re.Match) -> str:
        w_str, f_str = m.group(1), m.group(2)
        whole = int(w_str)
        leading_zeros = len(f_str) - len(f_str.lstrip("0"))
        if leading_zeros > 0:
            zero_prefix = " ".join(["sıfır"] * leading_zeros)
            frac_val = f_str.lstrip("0")
            if frac_val:
                frac_words = f"{zero_prefix} {int_to_turkish(int(frac_val))}"
            else:
                frac_words = zero_prefix
        else:
            frac_words = int_to_turkish(int(f_str))
        return f"{int_to_turkish(whole)} virgül {frac_words}"

    text = re.sub(r"\b(\d+),(\d+)\b", repl_decimal_comma, text)

    # 15. Decimal dots (non-thousands): 3.14 -> üç nokta on dört
    def repl_decimal_dot(m: re.Match) -> str:
        w_str, f_str = m.group(1), m.group(2)
        whole = int(w_str)
        leading_zeros = len(f_str) - len(f_str.lstrip("0"))
        if leading_zeros > 0:
            zero_prefix = " ".join(["sıfır"] * leading_zeros)
            frac_val = f_str.lstrip("0")
            if frac_val:
                frac_words = f"{zero_prefix} {int_to_turkish(int(frac_val))}"
            else:
                frac_words = zero_prefix
        else:
            frac_words = int_to_turkish(int(f_str))
        return f"{int_to_turkish(whole)} nokta {frac_words}"

    text = re.sub(r"\b(\d+)\.(\d+)\b", repl_decimal_dot, text)

    # 16. Standalone numbers with suffixes: 30'da, 30’a -> otuz'da, otuz'a
    def repl_suffix(m: re.Match) -> str:
        num = int(m.group(1))
        apostrophe = m.group(2)
        suffix = m.group(3)
        return f"{int_to_turkish(num)}{apostrophe}{suffix}"

    text = re.sub(r"\b(\d+)(['\u2019])([a-zA-ZçğıöşüÇĞİÖŞÜ]+)\b", repl_suffix, text)

    # 17. Negative integers: -5 -> eksi beş
    def repl_negative(m: re.Match) -> str:
        num = int(m.group(1))
        return f"eksi {int_to_turkish(num)}"

    text = re.sub(r"(?:^|(?<=\s))-(\d+)\b", repl_negative, text)

    # 18. Remaining standalone integers: 30 -> otuz, 123 -> yüz yirmi üç, 2026 -> iki bin yirmi altı
    def repl_int(m: re.Match) -> str:
        s = m.group(0)
        if (s.startswith("0") and len(s) > 1) or len(s) > 12:
            return digits_to_turkish(s)
        return int_to_turkish(int(s))

    text = re.sub(r"\b\d+\b", repl_int, text)

    # 19. Restore protected URLs, emails, and handles
    for ph, orig in placeholders.items():
        text = text.replace(ph, orig)

    return text


# =============================================================================
# 5. English Semantic Normalization
# =============================================================================

def normalize_english_numbers(text: str) -> str:
    """Normalize numeric expressions in English text into natural conversational words."""
    placeholders: dict[str, str] = {}
    p_counter = 0

    def protect_match(m: re.Match) -> str:
        nonlocal p_counter
        ph = f"__MICORA_PROT_{p_counter}__"
        placeholders[ph] = m.group(0)
        p_counter += 1
        return ph

    text = re.sub(r"https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.(?:com|org|net|io|dev|edu|gov)(?:/\S*)?", protect_match, text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", protect_match, text)
    text = re.sub(r"@[a-zA-Z0-9_]+", protect_match, text)

    # IP addresses
    def repl_ip(m: re.Match) -> str:
        parts = m.group(0).split(".")
        return " point ".join(int_to_english(int(p)) for p in parts)

    text = re.sub(r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b", repl_ip, text)

    # Versions
    def repl_version_prefixed(m: re.Match) -> str:
        prefix = m.group(1)
        ver_nums = m.group(2).split(".")
        ver_words = " point ".join(int_to_english(int(v)) for v in ver_nums)
        return f"{prefix} {ver_words}"

    text = re.sub(r"(?i)\b(v|ver|version)\s*(\d+(?:\.\d+)+)\b", repl_version_prefixed, text)

    def repl_version_standalone(m: re.Match) -> str:
        ver_nums = m.group(1).split(".")
        return " point ".join(int_to_english(int(v)) for v in ver_nums)

    text = re.sub(r"\b(\d+\.\d+\.\d+(?:\.\d+)*)\b", repl_version_standalone, text)

    # Currencies: $250.50, $250, $0.50, €100, £50, 50$, 100€
    def parse_curr_en(raw_num: str, curr: str) -> str:
        if "," in raw_num and "." not in raw_num:
            whole = int(raw_num.replace(",", ""))
            frac = 0
            frac_len = 0
        elif "." in raw_num:
            parts = raw_num.split(".")
            whole = int(parts[0].replace(",", ""))
            frac = int(parts[1])
            frac_len = len(parts[1])
        else:
            whole = int(raw_num)
            frac = 0
            frac_len = 0
        return format_currency_english(whole, frac, frac_len, curr)

    def repl_curr_en_pre(m: re.Match) -> str:
        return parse_curr_en(m.group(2), m.group(1))

    text = re.sub(r"(\$|€|£)\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\b", repl_curr_en_pre, text)

    def repl_curr_en_post(m: re.Match) -> str:
        return parse_curr_en(m.group(1), m.group(2))

    text = re.sub(r"\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(\$|€|£|USD|EUR|GBP)(?=$|[^\w])", repl_curr_en_post, text)

    # Verification codes / OTP / PIN / IDs / Serial numbers: code: 582910, pin: 4829, code is 123456, serial no: 948271
    def repl_otp_en(m: re.Match) -> str:
        kw = m.group(1)
        sep = m.group(2)
        digits = m.group(3).replace(" ", "")
        return f"{kw}{sep}{digits_to_english(digits)}"

    text = re.sub(
        r"(?i)\b(code|pin|otp|verification\s*code|serial\s*no|serial\s*number|order\s*id|user\s*id|tracking\s*number|account\s*number)([:\s]+|\s+is\s+)(\d{3,16}|\d{3,4}(?:\s+\d{3,4})+)\b",
        repl_otp_en,
        text,
    )

    # Percentages: 20%, 20.5%
    def repl_percent(m: re.Match) -> str:
        raw_num = m.group(1)
        if "." in raw_num:
            w, f = raw_num.split(".")
            val_str = f"{int_to_english(int(w))} point {int_to_english(int(f))}"
        else:
            val_str = int_to_english(int(raw_num))
        return f"{val_str} percent"

    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", repl_percent, text)

    # Units, physical quantities, durations, and measurements
    def repl_units_en(m: re.Match) -> str:
        return format_quantity_with_unit_english(m.group(1), m.group(2))

    text = re.sub(
        r"(-?\d+(?:\.\d+)?)\s*(GB|gb|MB|mb|KB|kb|TB|tb|kg|kilograms?|grams?|mg|tons?|km|kilometers?|meters?|cm|mm|liters?|litres?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|°C|°\s*C|degrees?)\b",
        repl_units_en,
        text,
    )
    text = re.sub(r"(-?\d+(?:\.\d+)?)\s+([ml])\b", repl_units_en, text)

    # Explicit separated digits: 1 2 3, 9 8 7 6
    def repl_sep_digits_en(m: re.Match) -> str:
        raw = m.group(0)
        digits = [c for c in raw if c.isdigit()]
        return " ".join(digits_to_english(d) for d in digits)

    text = re.sub(r"\b\d(?:\s+\d)+\b", repl_sep_digits_en, text)
    text = re.sub(r"\b\d(?:-\d){2,}\b", repl_sep_digits_en, text)

    # Time: 14:30 -> fourteen thirty, 09:05 -> nine oh five
    def repl_time(m: re.Match) -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        if mn == 0:
            return int_to_english(h)
        elif mn < 10:
            return f"{int_to_english(h)} oh {int_to_english(mn)}"
        else:
            return f"{int_to_english(h)} {int_to_english(mn)}"

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", repl_time, text)

    # English thousands: 1,000,000 -> one million
    def repl_en_thousands(m: re.Match) -> str:
        raw_num = m.group(0).replace(",", "")
        return int_to_english(int(raw_num))

    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", repl_en_thousands, text)

    # Decimal dots: 3.14 -> three point one four
    def repl_decimal_dot(m: re.Match) -> str:
        whole = int(m.group(1))
        frac_str = m.group(2)
        frac_words = " ".join(digits_to_english(d) for d in frac_str)
        return f"{int_to_english(whole)} point {frac_words}"

    text = re.sub(r"\b(\d+)\.(\d+)\b", repl_decimal_dot, text)

    # Standalone integers
    def repl_int(m: re.Match) -> str:
        s = m.group(0)
        if (s.startswith("0") and len(s) > 1) or len(s) > 12:
            return digits_to_english(s)
        return int_to_english(int(s))

    text = re.sub(r"\b\d+\b", repl_int, text)

    for ph, orig in placeholders.items():
        text = text.replace(ph, orig)

    return text


# =============================================================================
# 6. Master Normalization Pipeline
# =============================================================================

def normalize_text_for_tts(text: str) -> str:
    """Master text normalization pipeline for TTS input.

    Conservative cleaning: preserves Turkish phonemes and prosodic punctuation,
    normalizes whitespace, and converts numeric expressions into spoken words.
    Guaranteed idempotent: passing already-normalized text returns the exact same string.
    """
    if not text:
        return ""

    cleaned = str(text).strip()
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch == " ")

    if is_english_text(cleaned):
        cleaned = normalize_english_numbers(cleaned)
    else:
        cleaned = normalize_turkish_numbers(cleaned)

    # Clean multiple spaces
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    return cleaned.strip()
