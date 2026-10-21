"""Local development server (not shipped)."""
import os

PORT = int(os.environ.get("DEV_PORT", "5000"))

if __name__ == "__main__":
    print("dev server on :%d" % PORT)
