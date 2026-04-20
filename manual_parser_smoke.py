from app import parse_survival_records


def assert_ok(text: str) -> None:
    rows = parse_survival_records(text, "Group A")
    assert rows, f"Expected rows for input: {text!r}"


# Basic documented formats
assert_ok("5,1\n8,0\n11,0")
assert_ok("5, 1\n8, 0\n11, 0")
assert_ok("5 1\n8 0\n11 0")

# Full-width / CJK commas and spaces
assert_ok("5，1\n8，0\n11，0")
assert_ok("5， 1\n8， 0\n11， 0")
assert_ok("5　1\n8　0\n11　0")

# NBSP / tabs / zero-width chars
nbsp = "\u00A0"
zwsp = "\u200B"
bom = "\uFEFF"
assert_ok(f"5,{nbsp}1\n8\t0\n11{zwsp},{bom}0")

# Blank lines between records
assert_ok("\n5,1\n\n8,0\n\n11,0\n")

print("manual_parser_smoke: ok")
