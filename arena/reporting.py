import csv
import io


def to_csv(rows):
    if not rows:
        return ""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # Prevent model/run text from being interpreted as spreadsheet formulas.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                         for k, v in row.items()})
    return out.getvalue()


def _cell(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def _table(rows):
    if not rows:
        return "No data."
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(_cell(row[col]) for col in columns) + " |" for row in rows)
    return "\n".join(lines)


def to_markdown(comparison):
    warnings = "\n".join("- " + _cell(w) for w in comparison["warnings"])
    return (f"# Model Arena run comparison\n\nBaseline: {_cell(comparison['baseline'])}\n\n"
            f"Current: {_cell(comparison['current'])}\n\n"
            "Delta = current − baseline. Scores are absolute differences, not relative percentages.\n\n"
            "## Interpretation\n\n" + warnings + "\n\n"
            "Metrics use only pairs where both responses succeeded and both metric values exist. "
            "Check failures and paired sample counts before interpreting changes. Zero memory samples "
            "are treated as unavailable. Original scoring formulas are unchanged.\n\n"
            "## Coverage and failures\n\n" + _table(comparison["summaries"]) + "\n\n"
            "## Paired metric means\n\n" + _table(comparison["metrics"]) + "\n")
