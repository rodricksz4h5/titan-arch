# Titan Arch — pitch & demo script

## The 30-second pitch

> Cities break in ways nobody can see until it's too late — one flooded
> underpass can choke half a city. Titan Arch takes a city's roads, finds the
> handful of junctions everything secretly depends on, scores how fragile the
> city is, and lets a planner *click a junction to fail it* and watch the city
> re-route in real time. It even prescribes the single best new road to build.
> And when there's no map — only a satellite image with roads hidden under trees
> — it reconstructs the network by healing the gaps, with 100% precision on our
> benchmark. All of it runs on free, open data.

## Why it's hard / why it's novel

- Most road-extraction work stops at a **broken pixel mask** — useless for
  routing. We turn it into a **connected, routable network** by healing the gaps
  trees and shadows leave behind.
- Most resilience work stops at a **pretty heat-map**. We give a single
  **Resilience Index** and an interactive **failure simulator** — and then go one
  step further and **recommend the repair** that helps most.

## The 2-minute live demo (script)

1. **Load a real city.** Type `Indiranagar, Bengaluru` → Load.
   *"This is the real road network, pulled live from OpenStreetMap, colored by
   how critical each road is."* (Toggle the **Satellite** basemap to show it sits
   on real imagery.)
2. **Resilience Index.** Point at `0.86 · robust` and the little collapse curve.
   *"This city is robust — it has alternate routes."*
3. **Break it (the wow moment).** Click the #1 gatekeeper.
   *"Close that one junction and city-wide efficiency drops, the network
   fragments, and this cross-town trip now detours 600 m"* (or is cut off).
4. **Prescribe the fix.** Click **Suggest the best fix**.
   *"It found the single new link that lifts resilience from 0.86 to 0.93 —
   actionable advice, not just diagnosis."* (Green dashed line on the map.)
5. **Reconstruct from a satellite mask.** Click **Reconstruction demo**.
   *"This is a road mask with tree-occlusion gaps — 19 disconnected fragments.
   Titan Arch heals it back to one network, 99% connected, and every healed link
   is a real road the trees hid — 100% precision on our benchmark."* (Toggle
   "show healed" off/on to show the before/after.)

## Headline numbers

- Real city analysed in ~7 s; failure simulation < 1 s per click.
- Repair recommender: Resilience Index 0.86 → 0.93 from one short link.
- Healing benchmark: pieces 19 → 2, largest area 17% → 99%, **precision 100%**,
  ~49% routing-efficiency recovery.

## What's free / what runs where

- Roads: OpenStreetMap (Overpass) · Imagery: Esri World Imagery · both keyless.
- The whole analysis + dashboard runs on a **normal laptop, no GPU**.
- The only GPU step (satellite → road mask) is a one-time **free Colab** run that
  exports a mask you upload to the app.

## Honest limitations (say these before a judge asks)

- The road-segmentation model is fine-tuned briefly on free Colab; accuracy
  scales with training time. The reconstruction *healing* is validated at 100%
  precision on synthetic ground truth; on real masks it inherits the model's
  quality.
- On-demand satellite→mask for arbitrary places would need a hosted GPU; for the
  demo we pre-generate masks. The OSM-based analysis needs no GPU ever.
