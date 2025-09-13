import sys
from pathlib import Path
import pandas as pd

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).strip() for c in df.columns})

def pick_col(df: pd.DataFrame, wanted: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    if wanted in df.columns:
        return wanted
    if wanted.lower() in lower:
        return lower[wanted.lower()]
    raise KeyError(f"找不到列 {wanted}；现有列：{list(df.columns)}")

def main(map_csv: str, right_csv: str, out_csv: str = "new_old_name_map_leftjoin.csv"):
    left = pd.read_csv(map_csv)
    left = normalize_cols(left)
    c_file = pick_col(left, "file_name")
    c_90   = pick_col(left, "90_file_name")

    right = pd.read_csv(right_csv)
    right = normalize_cols(right)
    r_file = pick_col(right, "file_name")

    

    # 左连接（只带上右表的键列，避免无谓的列扩展）
    right_cols = [r_file, "verbal_responsiveness", "ictal_vocalization"]
    merged = left.merge(right[right_cols], how="left", left_on=c_file, right_on=r_file)
    print(merged.head(5))
    # 只保留左表两列，并标准化列名
    out = merged[[c_file, c_90, "verbal_responsiveness", "ictal_vocalization", ]].copy()
    out.columns = ["file_name", "90_file_name", "verbal_responsiveness", "ictal_vocalization", "extra_col"]

    # 若右表存在重复键造成重复行，这里去重
    out = out.drop_duplicates(subset=["file_name", "90_file_name"])

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已保存：{out_csv}（{len(out)} 行）")

if __name__ == "__main__":
    # 用法：
    # python left_join_keep_left_cols.py new_old_name_map.csv Task1_AF3_Full_Results.csv [输出文件名]
    map_csv  = sys.argv[1] if len(sys.argv) > 1 else "new_old_name_map.csv"
    right_csv = sys.argv[2] if len(sys.argv) > 2 else "/mnt/SSD3/lina/ssb/vlm_inference/audio-flamingo-3/Task1_AF3_Full_Results.csv"
    out_csv   = sys.argv[3] if len(sys.argv) > 3 else "90_audio_feature.csv"
    main(map_csv, right_csv, out_csv)
