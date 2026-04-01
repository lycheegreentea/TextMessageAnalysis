
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SEP  = ";" if sys.platform == "win32" else ":"

subprocess.run([
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--name", "word2vec_explorer",
    f"--add-data={HERE / 'index.html'}{SEP}.",

    "--hidden-import=gensim",
    "--hidden-import=gensim.models.word2vec",
    "--hidden-import=gensim.models.keyedvectors",

    "--hidden-import=scipy.special.cython_special",
    "--hidden-import=scipy._lib.messagestream",

    "--hidden-import=jieba",
    "--hidden-import=jieba.analyse",

    "app.py"
], check=True)

print("\nBuild complete")
