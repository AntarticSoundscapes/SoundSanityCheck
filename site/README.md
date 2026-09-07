# Antarctic Wind Atlas

`index.html` is a single self-contained page: the aggregated numbers are inlined
into it, so it needs no server, no build step and no network beyond the Google
Fonts stylesheet. Open it directly, or drop it on any static host.

## Rebuilding it after a new analysis run

```sh
uv run python scripts/export_site_data.py --out site/data.json
```

That collapses the per-recording shards in `data/analysis/` into the site x
season x wind-band cells the page draws (~35 kB). Then re-inline them:

```sh
python3 - <<'PY'
import pathlib, re
page = pathlib.Path("site/index.html")
data = pathlib.Path("site/data.json").read_text()
html = page.read_text()
page.write_text(re.sub(
    r'(<script id="atlas-data" type="application/json">).*?(</script>)',
    lambda m: m.group(1) + data + m.group(2), html, flags=re.S))
PY
```

`data.json` is kept alongside as the readable form of what the page contains.