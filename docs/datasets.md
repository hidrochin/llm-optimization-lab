# Public datasets

Public data used in this repo, its license, and how it was built. Private data never
appears here; its counts and hashes go to `reports/data_quality.md` (T-L03).

## CORD v2: the dev set

Phase 1 validates every tool on CORD v2 before private data exists ([T-L01](../TASKS.md#t-l01--public-dev-set-cord-v2-)).
Dev runs are never results ([rule 7](../TASKS.md#rules-for-experiments-and-results-agreed-2026-11-20)).

| | |
|---|---|
| Source | [`naver-clova-ix/cord-v2`](https://huggingface.co/datasets/naver-clova-ix/cord-v2), revision `7f0115a4b758a71d6473b8d085751692da2fef98` |
| Content | 1,000 Indonesian shop and restaurant receipts: photo, OCR boxes, parse annotation |
| License | CC BY 4.0: internal and commercial use allowed with attribution; no other obligations |
| Paper | Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, Document Intelligence Workshop at NeurIPS 2019 |
| Location | `$LLMOPT_DATA_DIR/public/cord/`, 401 MB |
| Built | 2026-11-25 by the lead with `scripts/data/convert_cord.py` |

### Conversion

`scripts/data/convert_cord.py` writes two records per receipt sharing one `group_id`:
`-img` (the photo, re-encoded as JPEG quality 95) and `-txt` (spatial text rendered from
the OCR boxes). The target follows [`doc_types/cord_receipt/schema.json`](../doc_types/cord_receipt/schema.json):

- `menu[]`: `name`, `count`, `unit_price`, `price`; `subtotal`: subtotal, discount,
  service, tax; `total`: total, cash, change, credit card. Every key is always present,
  `null` when not annotated. Values are kept as printed (`"1 x"`, `"45,500"`).
- Dropped CORD keys (counts on the 800 CORD train receipts): `total.menuqty_cnt` (228),
  `menu.sub` sub-items (183 items), `menu.num` (93), `menu.discountprice` (86),
  `sub_total.etc` (68), `total.emoneyprice` (46), `total.menutype_cnt` (42), and a few
  rarer ones. The text of dropped values is still in the image and the spatial text.
- A section annotated twice (one receipt, CORD train 692: a subtotal before and after a
  discount) is merged; for each key, the first block with a value wins.

**Caveat:** CORD's OCR boxes are ground truth, so the spatial text has no OCR errors.
Spatial-text scores on CORD are optimistic compared with production OCR.

### Splits

The official CORD split (800 / 100 / 100) is not kept: records are re-split with
`scripts/split_dataset.py` (seed 42, default ratios 80 / 5 / 10 / 5), by receipt, so the
image and spatial text of one receipt are always in the same split. Scores are therefore
not comparable with published CORD results.

| Split | Receipts | Records (image + spatial text) |
|-------|---------:|-------------------------------:|
| train | 800 | 1,600 |
| validation | 50 | 100 |
| test | 100 | 200 |
| calibration | 50 | 100 |
| benchmark (from test) | 50 | 100 |

Targets: 1 to 22 menu items per receipt (median 2). Null rate per field, over all
receipts: `total.total_price` 3%, `menu[].price` 1%, `menu[].count` 10%,
`menu[].unit_price` 71%, `subtotal.subtotal_price` 34%, `subtotal.tax_price` 56%,
`subtotal.discount_price` 93%, `total.credit_card_price` 85%. Spatial text is
43–1,141 characters (median 224), so CORD does not exercise long context.

SHA-256 of the record files:

```
837eb03ca62d723bcebeb2cbb9a089eb157f099f4639ddd6a0f05ced1d12f395  all.jsonl
bc6b399074442ebcf913f7b3d766063ff22f6a9e90bbeb76cd9f558dcc0bc025  train.jsonl
55a73992767dab826bbbf878f7dfde477d78caec4f2d46e8c1519d3def5453f3  validation.jsonl
d20722a7e88cdad9e461915ea83eecaab10b57ab910ed5e11acf4750ddfed3f7  test.jsonl
7b7e8258fa22357cf79611f0e63db53c7aa09b58b979fc13f696871959aaf734  calibration.jsonl
59518631dd8eea671a82acdd59897fdf7fe150a5fef194dbfc5c935660853f20  benchmark.jsonl
```

### Copy to a GPU server

Copy the built folder instead of converting again on each server: JPEG re-encoding
depends on the Pillow version, so a second conversion can give different image bytes.
`SHA256SUMS` in the folder covers every file, images included.

```bash
rsync -a "$LLMOPT_DATA_DIR/public/cord/" <server>:<data dir>/public/cord/
ssh <server> 'cd <data dir>/public/cord && sha256sum -c --quiet SHA256SUMS && echo OK'
```

## CORD v2 with every field: `cord_full`

The field-holdout experiment E20 ([T-L15](../TASKS.md#t-l15--e20-field-holdout-fixed-vs-sampled-requests-)) needs
fields that training never sees but the test set labels. `public/cord_full/` is the dev
set above with every CORD field in the target ([`doc_types/cord_receipt_full`](../doc_types/cord_receipt_full/schema.json), 27 fields):

```bash
uv run --group data python scripts/data/convert_cord.py \
    --out-dir "$LLMOPT_DATA_DIR/public/cord_full" --relabel-from "$LLMOPT_DATA_DIR/public/cord"
```

- Same record ids, split files, spatial text and images as `public/cord` (`images` is a
  relative link to `../cord/images`), so the frozen split holds. Only `doc_type`, `target`
  and `label_source: human` differ. Built 2026-11-26 by the lead, 5.7 MB.
- Added fields: `menu[]` `num`, `discount_price`, `item_subtotal`, `vat_flag`, `etc` and
  `sub[]` (`name`, `count`, `unit_price`, `price`); `subtotal.other_service_price`,
  `subtotal.etc`; `total` `total_etc`, `emoney_price`, `menu_type_count`,
  `menu_quantity_count`. `void_menu` (one receipt) is left out.
- Test receipts (of 100) with a value, for the fields E20 holds out: `total.change_price`
  64, `subtotal.tax_price` 36, `menu[].unit_price` 30, `total.menu_quantity_count` 29.

`SHA256SUMS` in the folder covers its record files:

```
b7e368efdebd4cd54e9f26069048f051577e12fb8efb849be3fd40e7890d5052  all.jsonl
0d26aa504570165c18b912f1beae8c25583fc4eef342bc6d794245d5bb968f2b  train.jsonl
aaf2ebe0b8c807a1198ad620d31dfc4550f32d9c97b4633ef2bb26ee1225ad4c  validation.jsonl
9b8d03d850bfe2f2117d4101c2c09b3912581de20f0aec4fa14ae787a3635cff  test.jsonl
feaf8ad62bee227c745fa27f428232aff80e1b42d5576e1b10b8f71bcc11c130  calibration.jsonl
2d98dfe5e86ef08a20a4dbc90243c5f8cc7e56bdfe70ec72f52679499629d944  benchmark.jsonl
47ef2844bb87559868ac2e9413620d41eee686e7895fc3a7b74e76fa92122441  split_manifest.json
```

Copy it next to `public/cord` (the image link is relative): `rsync -a` keeps the link.
