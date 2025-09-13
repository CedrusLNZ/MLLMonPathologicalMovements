import sys
from pathlib import Path
import pandas as pd

def load_table(path: Path) -> pd.DataFrame:
    # 读入
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}

    def pick(col_key: str) -> str:
        if col_key in df.columns:
            return col_key
        if col_key.lower() in lower_map:
            return lower_map[col_key.lower()]
        raise KeyError(f"找不到列 {col_key}。可用列有：{list(df.columns)}")

    c1 = pick("file_name")
    c2 = pick("90_file_name")


    out = df[[c1, c2]].copy()
    out.columns = ["file_name", "90_file_name"]

  
    mask = out["90_file_name"].notna() & out["90_file_name"].astype(str).str.strip().ne("")
    out = out[mask]

    return out

def main(input_path: str, output_path: str = "new_old_name_map.csv"):
    in_path = Path(input_path)
    out_df = load_table(in_path)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已保存：{output_path}（{len(out_df)} 行）")

if __name__ == "__main__":
    # 用法：
    # python script.py your_file.csv
    # 或
    # python script.py your_file.xlsx new_old_name_map.csv
    input_path = sys.argv[1] if len(sys.argv) > 1 else "final_video_df.xlsx"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "new_old_name_map.csv"
    main(input_path, output_path)
