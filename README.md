# JustSign

A rhythm game where letters fall like Guitar Hero notes, and instead of pressing
frets you fingerspell each letter in ASL as it crosses the hit circle.

## Running it

This is a plain static site — no build step.

1. Download all three files (`index.html`, `style.css`, `script.js`) into the same folder.
2. Serve them over `http://localhost` or `https://` — camera access won't work from a
   bare `file://` path in most browsers. Easiest option:
   ```
   npx serve .
   ```
   or `python3 -m http.server`, then open the printed `localhost` URL.
3. Allow camera access when prompted. The hand-tracking model (~10MB) loads from
   Google's CDN on first run.

## About the two Google tools you mentioned

Quick clarification so the setup makes sense:

- **Google Stitch** is a design tool that *generates UI mockups* from a prompt — it's not
  a library a website can import at runtime, so there's nothing to "install." I designed
  the interface directly (see the "signature element" note below), but if you want, you
  could run this brief through Stitch to get an alternate visual direction, then hand me
  the result to rebuild the CSS around it.
- **Google Antigravity** is Google's agentic coding IDE (an environment for building
  software), not an API — same story, nothing to embed in a webpage.

Where the **Gemini API** genuinely fits, I used it:
- **Themed word lists** — paste an AI Studio key in on the start screen, hit "🎲 Themed
  words," give it a theme, and Gemini generates a word list (filtered to only use letters
  the game supports) for Word Mode.
- **Post-game commentary** — after a run, if a key is present, Gemini writes a couple of
  lines reacting to your score/accuracy/combo.

I did **not** call Gemini per-frame to grade your hand sign — that's a latency and cost
problem for something that needs to run at 30-60fps. Live scoring uses a small
hand-written classifier on top of MediaPipe's landmark output instead, per your note
that "score should be done using code for now." If you later want more accurate/complete
ASL recognition, swapping in a proper trained classifier (or batching snapshots to Gemini
Vision for asynchronous feedback, not live judging) would be the next step.

## Supported letters

Real fingerspelling recognition from simple geometric rules is hard to get right for the
full alphabet — several letters (M, N, S, T, E, K, P, Q, R, X, G, H) differ only in subtle
thumb/finger position, and J/Z require motion, not a static pose. To keep detection honest
rather than flaky, the game currently supports a subset with clearly distinct handshapes:

```
A B C D F I L O U V W Y
```

Word Mode's built-in word list and Gemini's generated words are restricted to these
letters. Extending the letter set means adding an entry to the `SIGNS` object in
`script.js` (target finger-curl values + thumb/index distance) — the diagram, classifier,
and word filtering all read from that one place.

## Notes on accuracy

Lighting, camera angle, and hand size all affect the heuristic classifier. If it feels
too strict or too loose, the easiest dial is the `confidence > 0.35` threshold in
`updateEngine()` inside `script.js`, and the per-letter feature values in `SIGNS`.
