#!/usr/bin/env bash
# Hard-constraint gate (§1): the artificial-decipherment validation is void if
# any component may have seen relevant scholarship during pretraining. Fails
# the build if pretrained-weight loading appears anywhere outside
# contrib/openbook/ (the only directory where it will ever be permitted).
#
# Pretrained DATASETS are allowed; pretrained MODELS are not.
set -euo pipefail

root="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

pattern='from_pretrained|AutoModel|AutoTokenizer|AutoConfig|torch\.hub|hf_hub_download|snapshot_download|transformers\.pipeline|timm\.create_model'

matches=$(grep -rnE "$pattern" \
    --include='*.py' \
    "$root/glyphos" "$root/scripts" "$root/tests" 2>/dev/null \
    | grep -v 'contrib/openbook' || true)

if [ -n "$matches" ]; then
    echo "FORBIDDEN pretrained-weight usage found (hard constraint, CONVENTIONS.md §1):"
    echo "$matches"
    exit 1
fi
echo "no-pretrained check: clean"
