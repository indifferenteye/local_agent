#!/usr/bin/env python3
import os
import json

def list_files(directory="."):
    try:
        files = os.listdir(directory)
        return {"success": True, "files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = list_files()
    print(json.dumps(result))