#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append("/usr/lib/python3/dist-packages")

import uno  # type: ignore
from com.sun.star.beans import PropertyValue  # type: ignore


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def main(path: str):
    target = Path(path).resolve()
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    ctx = resolver.resolve(
        "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", ctx
    )
    url = uno.systemPathToFileUrl(str(target))
    doc = desktop.loadComponentFromURL(url, "_blank", 0, (prop("Hidden", True),))
    for idx in doc.getDocumentIndexes():
        idx.update()
    doc.getTextFields().refresh()
    doc.store()
    doc.close(True)
    print(target)


if __name__ == "__main__":
    main(sys.argv[1])
