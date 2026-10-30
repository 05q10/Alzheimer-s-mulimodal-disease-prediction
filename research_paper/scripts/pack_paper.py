import zipfile
import shutil
from pathlib import Path

def pack():
    base_dir = Path("research_paper")
    zip_dest = Path("research_paper_v2.zip")
    
    files_to_zip = []
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base_dir)
        parts = rel.parts
        # Exclude directories
        if any(ignored in parts for ignored in ["previous", "__pycache__", ".git", ".idea", ".vscode"]):
            continue
        # Exclude extensions
        if p.suffix in [".aux", ".log", ".out", ".toc", ".synctex.gz", ".bbl", ".blg", ".zip"]:
            continue
        files_to_zip.append((p, rel))
        
    print(f"Adding {len(files_to_zip)} files to {zip_dest.name}...")
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files_to_zip:
            zf.write(src, arcname)
            
    print(f"Created {zip_dest} ({zip_dest.stat().st_size / (1024*1024):.2f} MB)")
    
    # Also create copy inside research_paper/
    dest_in_dir = base_dir / "research_paper_v2.zip"
    shutil.copy2(zip_dest, dest_in_dir)
    print(f"Copied to {dest_in_dir}")

if __name__ == "__main__":
    pack()
