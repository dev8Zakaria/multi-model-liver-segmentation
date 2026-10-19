"""Replace Git LFS pointer files in models/ by the real ONNX files (Docker build step).

Some hosts clone the repository without Git LFS, leaving ~130-byte pointer
files instead of the models. Each pointer is downloaded from MODEL_BASE_URL
and checked against the sha256 recorded in the pointer. Real files are left
untouched, so the step is harmless when LFS was fetched.
"""
import hashlib
import os
import sys
import urllib.request

MODELS_DIR = sys.argv[1] if len(sys.argv) > 1 else "models"
BASE_URL = os.environ.get(
    "MODEL_BASE_URL",
    "https://media.githubusercontent.com/media/dev8Zakaria/multi-model-liver-segmentation/main/models/",
)
POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def read_pointer(path):
    with open(path, "rb") as f:
        head = f.read(512)
    if not head.startswith(POINTER_PREFIX):
        return None
    fields = dict(line.split(" ", 1) for line in head.decode().splitlines() if " " in line)
    return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def main():
    for name in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, name)
        pointer = read_pointer(path)
        if pointer is None:
            print(f"{name}: real file, kept")
            continue
        oid, size = pointer
        print(f"{name}: LFS pointer, downloading {size / 1e6:.1f} MB")
        digest = hashlib.sha256()
        tmp = path + ".part"
        with urllib.request.urlopen(BASE_URL + name) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != oid:
            os.remove(tmp)
            sys.exit(f"{name}: checksum mismatch, download corrupted")
        os.replace(tmp, path)


if __name__ == "__main__":
    main()
