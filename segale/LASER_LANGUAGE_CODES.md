# LASER/NLLB Language Codes

LASER uses NLLB (No Language Left Behind) language codes in the format: `{language_code}_{script_code}`

## Common Language Codes

Based on your experiments, here are the relevant codes:

| Your Code | LASER/NLLB Code | Language | Script |
|-----------|----------------|----------|--------|
| `en` | `eng_Latn` | English | Latin |
| `zht` | `zho_Hant` | Traditional Chinese | Han (Traditional) |
| `zh` | `zho_Hans` | Simplified Chinese | Han (Simplified) |
| `es` | `spa_Latn` | Spanish | Latin |
| `de` | `deu_Latn` | German | Latin |
| `fr` | `fra_Latn` | French | Latin |
| `it` | `ita_Latn` | Italian | Latin |
| `ru` | `rus_Cyrl` | Russian | Cyrillic |
| `ko` | `kor_Hang` | Korean | Hangul |
| `vi` | `vie_Latn` | Vietnamese | Latin |

## Downloading Models

To download models for your languages:

```bash
cd ~/Documents/Code/LASER

# Single language
bash ./nllb/download_models.sh zho_Hant

# Multiple languages (space-separated)
bash ./nllb/download_models.sh zho_Hant eng_Latn spa_Latn

# All common languages for your experiments
bash ./nllb/download_models.sh eng_Latn zho_Hant zho_Hans spa_Latn deu_Latn fra_Latn ita_Latn rus_Cyrl kor_Hang vie_Latn
```

## Note

- `ace_Latn` in the original documentation was just an example (Acehnese in Latin script)
- You need to download models for the languages you're actually testing
- Traditional Chinese (`zht`) requires `zho_Hant`, not `ace_Latn`
- Models can be large, so only download what you need

## Finding Other Language Codes

If you need other languages, check:
- NLLB documentation: https://github.com/facebookresearch/fairseq/tree/nllb
- FLORES-200 language codes: https://github.com/facebookresearch/flores

