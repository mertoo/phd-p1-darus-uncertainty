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


def test_tex_minus_leaves_ranges_and_labels():
    assert report_pub.tex_minus("AR-SCP & -0.51 [-1.2, 0.3] & 75.2--77.7") == "AR-SCP & $-$0.51 [$-$1.2, 0.3] & 75.2--77.7"


def test_backbone_prose_values_match_table(tmp_path):
    """Analysis 2.4: the OOD coverage range quoted for GRU/TCN/linear is the range of the corresponding table cells."""
    report_pub.table_backbones(str(tmp_path))
    report_pub.prose_values_backbones(str(tmp_path))
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (tmp_path / "prose_values_backbones.tex").read_text()))
    rows = [ln.split("&") for ln in (tmp_path / "tab_backbones.tex").read_text().splitlines()
            if ln.startswith(("EN-SCP", "DN-SCP")) and any(b in ln for b in ("GRU", "TCN", "Linear"))]
    cov = [float(r[5]) for r in rows]
    assert macros["BBcovSpreadOODnew"] == f"{min(cov):.1f}--{max(cov):.1f}"
    assert int(macros["BBnISlowerWM"]) <= int(macros["BBnPairs"]) == 9
