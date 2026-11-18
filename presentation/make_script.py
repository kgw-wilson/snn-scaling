"""Writes script.md from notes.json (emitted by build_deck.js)."""
import json

notes = json.load(open('notes.json'))
main = [x for x in notes if not x['title'].startswith('Backup')]
total = sum(x['minutes'] for x in main)
cut = [x for x in main if 'CUT IF LONG' in x['title']]
L = ["# Speaker script: spiking-network execution strategies on CPU and GPU", "",
 f"Slides 1–25 are planned at **{total:g} min**; with transitions that comes to about 45 min, then 15 min of questions.", "",
 "**If you run long**, cut these slides (marked [CUT IF LONG]): " + ", ".join(f"{x['num']} ({x['title'].replace(' [CUT IF LONG]','')}, {x['minutes']:g} min)" for x in cut)
 + f". Together they save about {sum(x['minutes'] for x in cut):g} min.", "",
 "The same text is in each slide's speaker notes. \"If asked\" lines are prompts for likely questions.", ""]
el = 0
for x in notes:
    if x['title'].startswith('Backup'):
        if x['num'] == main[-1]['num'] + 1:
            L += ["---", "", "## Backup slides (Q&A only)", ""]
        L += [f"### Slide {x['num']}: {x['title'].replace('Backup: ','')}", "", x['body'], ""]
        continue
    el += x['minutes']
    L += [f"## Slide {x['num']}: {x['title']} ({x['minutes']:g} min, elapsed {el:g})", "", x['body'], ""]
open('script.md', 'w').write("\n".join(L))
