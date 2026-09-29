BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def fmt_angka(x, desimal=2, signed=False):
    teks = f"{x:{'+' if signed else ''},.{desimal}f}"
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_rp(x, desimal=2):
    return f"Rp {fmt_angka(x, desimal)}"


def fmt_persen(x, desimal=2):
    return f"{fmt_angka(x, desimal)}%"


def fmt_tanggal(ts):
    return f"{ts.day} {BULAN[ts.month - 1]} {ts.year}"