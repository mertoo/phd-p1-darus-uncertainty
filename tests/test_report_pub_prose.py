import re

from src.analysis import report_pub


def test_prose_values_match_generated_table(tmp_path):
    """Numbers quoted in the text come from the same outputs as Table 4 (AR-SCP ID widths, rounded alike)."""
    S = report_pub.summaries()
    report_pub.table_uq(S, str(tmp_path), "test", "tab_uq_id")
    report_pub.prose_values(S, str(tmp_path))
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (tmp_path / "prose_values.tex").read_text()))
    rows = [ln for ln in (tmp_path / "tab_uq_id.tex").read_text().splitlines()
            if ln.startswith("AR-SCP") and "Naive" not in ln]
    widths = [float(ln.split("&")[3].split("[")[0]) for ln in rows]
    assert macros["ARwidthID"] == f"{min(widths):.2f}--{max(widths):.2f}"
    assert float(macros["SeedRangeRawGauss"]) > float(macros["CIspanRawGauss"].split("--")[-1])
