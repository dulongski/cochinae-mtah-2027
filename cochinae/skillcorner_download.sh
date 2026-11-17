#!/usr/bin/env bash
# Download SkillCorner open data (MIT, credit SkillCorner + PySport) pinned to one commit.
#   ./cochinae/skillcorner_download.sh [dest]      default: $MTAH_DATA_DIR/skillcorner
# Tracking files are checked against the size in their Git LFS pointer; incomplete ones are
# downloaded again (to a .part file, moved into place when complete). 6 matches in parallel.
set -euo pipefail
REV=4340d274572876239c154c90bc507a9b3250a656
DEST="${1:-${MTAH_DATA_DIR:-data}/skillcorner}"
export RAW=https://raw.githubusercontent.com/SkillCorner/opendata/$REV/data
export LFS=https://media.githubusercontent.com/media/SkillCorner/opendata/$REV/data
export DEST
mkdir -p "$DEST"
curl -sfL -o "$DEST/matches.json" "$RAW/matches.json"
one() {
  id=$1; d="$DEST/$id"; mkdir -p "$d"
  for f in match.json phases_of_play.csv; do
    [ -s "$d/${id}_$f" ] || curl -sfL -o "$d/${id}_$f" "$RAW/matches/$id/${id}_$f"
  done
  name="${id}_tracking_extrapolated.jsonl"
  want=$(curl -sfL "$RAW/matches/$id/$name" | awk '/^size/{print $2}')
  have=$(stat -f %z "$d/$name" 2>/dev/null || echo 0)
  if [ "$have" != "$want" ]; then
    curl -sfL -o "$d/$name.part" "$LFS/matches/$id/$name?download=1"
    mv -f "$d/$name.part" "$d/$name"
  fi
  echo "match $id done"
}
export -f one
python3 -c "import json,sys;print('\n'.join(str(m['id']) for m in json.load(open(sys.argv[1]))))" "$DEST/matches.json" \
  | xargs -P 6 -I{} bash -c 'one {}'
echo "done: all matches"
