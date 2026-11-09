# OpenNeural

A 3D simulator with a real trained neural driver, browser training, and held-out evaluation. Models, scenery, libraries, and weights are bundled. **Alpine Circuit is used for training; Canyon Run is held out.**

## Run

Requires Node.js 18+. No npm install is needed.

```sh
git clone https://github.com/asharaamer0/OpenNeural.git
cd OpenNeural
npm run dev
```

Open http://127.0.0.1:4173. The `dist/` directory also works on any static HTTP server. File URLs cannot load ES modules. Optional Google Fonts fall back to system fonts.

## Road — multi-agent traffic

Open **Road** in the environment sidebar, or `/road.html`. A four-lane boulevard loops around OpenNeural Square, with two lanes in each direction, pavements, shops, park paths, trees, four signal-controlled zebra crossings, 18 pedestrians, and two dogs. Select 8, 12, 16, or 24 cars. All-neural mode gives every car an independent observation and neural forward pass using shared weights; the scripted expert is a separately labelled comparison mode. Click any fleet card or a car in the scene for a chase camera, then select **Driver POV**, or return to **Overview** to watch the whole road. Space pauses, C cycles cameras, R resets, Escape pauses. Backgrounding the page pauses traffic.

The **Road** policy is separate from the circuit model: 18 → 64 → 64 → 2 (5,506 parameters), with tanh hidden layers and steering/signed-pedal outputs. It observes local geometry, lane/heading error, speed and wheel state, desired speed, lead distance/speed, stopping requirement/distance, pedestrian conflict, and physical stopping/headway margins. It uses perfect simulator state and mapped geometry. Vehicles integrate a Frenet bicycle model at 60 Hz; simultaneous per-car inference occurs at 20 Hz. Cars are not replaying a trajectory. Signals and walking actors are scripted. Collision detection records body overlaps; it does not resolve rigid-body crashes. A contact pauses the live simulation by default and remains counted.

The explicit rules are keep left, remain in an assigned lane, stay below 50 km/h, follow lead vehicles with room to stop, stop at red (and amber when stopping is feasible), and yield to crossing people and dogs. **Dog crossing** adds an unexpected crossing ahead of the selected car. There are no road junctions, lane changes, overtaking, or emergency vehicles yet. These rules do not represent every driving law or certify morality or real-world safety.

The optional **braking supervisor** is a separate stopping/following envelope. Predicted versus applied pedal commands and intervention episodes are visible; steering always comes from the selected controller. Disable it to watch neural-only behavior. Violation/contact counts are never cleared by assistance. The live display and session export report contacts, red crossings, lane departure (>0.85 m) and speeding (>50 km/h plus 0.3 m/s tolerance) time, unsafe headway (<max(2 m, 0.8 × speed)) time while moving, yields, intervention episodes, and fleet distance. The score is a heuristic penalty score, not a safety probability. Counters restart only on an explicit reset, settings change, or model replacement.

The bundled policy was trained from random initialization with PyTorch/Adam for 180 epochs. Its 88,400 expert-labelled examples combine 50,000 perturbed traffic states with 120-second, 16-car expert rollouts using seeds 31 and 73 (10 Hz sampling). There are 75,243 training / 13,157 validation examples. Random frame splitting makes this an interpolation validation set; adjacent rollout frames may cross the split. The separate closed-loop tests below are the driving evidence. Best validation MSE is **0.003637**. Training loss weights steering 1.5×; reported validation MSE is unweighted.

Bundled closed-loop evaluation runs nine cases for 180 seconds each: 12 cars/seed 42, 16/73, 24/91, and 16/731 with an extra dog at 30 seconds, each with and without the supervisor, plus one unassisted expert comparison. All **9/9** passed with zero vehicle/vulnerable-user contacts, red crossings, lane departures, speeding, or unsafe headway time. The four neural-only runs covered 18.51, 24.91, 36.66, and 24.66 fleet km, respectively, with zero guard interventions. Assisted runs used 54, 43, 234, and 39 intervention episodes; these are not counted as neural-only success. A pass also requires >500 fleet meters, so a parked fleet cannot pass. Results establish performance only for these deterministic scenarios. Inspect/export all rows in the lab or `dist/models/road-evaluation.json`.

**Continue training** runs 20 real TensorFlow.js/Adam epochs in a worker, with 6,000 new expert states plus two 40-second queue rollouts (18,800 examples). It keeps current normalization and retains the best validation checkpoint, including the initial weights if training does not improve them. After training, nine frozen-policy cases run automatically; results include failures. Training/evaluation pauses live traffic. Cancel terminates the job and preserves the old model. Export/import Road models, export measured evaluation, and export the live session. Road data is local to the browser; it does not update the circuit model or touch Harbor Sprint's blind challenge.

Reproduce Road training with Python, NumPy, and PyTorch:

```powershell
npm run road:data
npm run road:train
npm run road:evaluate
npm test
```

Training data goes to ignored `training/road-data.json`. Weights and measured reports go to `dist/models/road-policy.json` and `road-evaluation.json`. Training/data seeds are fixed; hardware and library differences may affect weights. Core dynamics, actor rules, pure neural inference, per-agent observations, collision counting, model validation, no-fallback behavior, and all nine fleet scenarios are covered by the tests.

## Use the neural driver

**Harbor Sprint** is a third, completely new coastal course for a live first-attempt challenge. It is excluded from training, automatic evaluation, and neural/reference driving tests. Geometry alone is checked for closure and crossings. Select it and drive manually, or click **First neural attempt**. The live attempt copies the current model weights before driving and keeps that copy frozen, even if another model becomes active. Attempts are counted per driver/model, including interrupted runs; logs persist locally and can be exported. The automatic 10-attempt suite still covers Alpine and Canyon only. Harbor neural attempts stop after a completed lap or fail on repeated contacts, prolonged off-track driving, a stall, or a five-minute timeout.

The bundled model was actually trained from random weights on Track 1. Open **Behavioral cloning** to inspect training curves and results. **Drive Track 2** starts a live neural lap on the held-out circuit. **Neural** mode works on either track and stops after a completed lap, reporting time, score, contacts, and off-track time. It predicts at 20 Hz under the same 120 Hz bicycle-model physics used during evaluation. Learned driving contains no reference-controller fallback, replay, or stored route actions.

**Train on Track 1** performs real TensorFlow.js/Adam training in a Web Worker. Choose continued weights or new random initialization (100 epochs recommended), epoch count, and expert or manual data. The best validation weights are retained, then evaluated with frozen weights. Training never uses Canyon Run examples to update weights. Cancel terminates the worker and preserves the previous model. Export/import models and export measured results as JSON. Browser training is CPU based; duration depends on your device. Models persist in localStorage when space permits.

## What is learned

An 8 → 48 → 48 → 2 multilayer perceptron with tanh hidden layers and linear outputs (2,882 learned parameters). Its eight observations are speed, actual wheel angle, lane offset, heading error, lateral/forward coordinates of a road lookahead point, maximum preview curvature, and near-road curvature. Inputs are standardized using training-data statistics only. Outputs are steering and a signed pedal command (positive throttle, negative brake), clipped to action limits. Phase 1 manual ramps and speed-sensitive steering are preserved.

Absolute coordinates, track ID, lap progress, and expert actions are excluded from neural inputs. Features are invariant to translating or rotating a track. **This model uses perfect simulated vehicle state and mapped road geometry, not RGB images.** A camera-only model needs a separate perception and image-data pipeline. This is a simulation research sandbox, not a real-vehicle control stack.

## Demonstrations and validation

The bundled dataset contains a clean expert lap plus expert-labelled position (±2.4 m), heading (±0.325 rad), speed (0–30 m/s), and wheel-angle perturbations on Track 1. Mirroring inverts signed features and steering to teach both turn directions. These are supervised recovery examples, not human recordings or DAgger. The full dataset has 38,080 examples: 32,718 training and 5,362 validation. Mirrored pairs remain in the same split. Neither training nor normalization uses the test track.

The browser generates a smaller fixed dataset: 6,000 perturbation states plus the clean expert lap and mirroring. Continued training uses the current model's normalization and weights. Random initialization learns new weights and normalization. Shuffle order can vary, so browser weights are not guaranteed identical.

Manual training accepts only current-physics (`1.1.0`) Track 1 **manual** recordings. It excludes old physics, Track 2, neural recordings, and handbrake frames. At least 300 usable original frames are required. Recordings are split into contiguous 100-frame groups; mirrored pairs stay together. Manual examples include your mistakes, so evaluate the result. Recordings include neural feature vectors, wheel angle, model ID for neural attempts, state, five road-edge sensors, and actions at 20 Hz. Record up to 15 minutes per drive, retain up to 12 drives, and export JSON. Storage failures leave data in memory and are reported; export before closing.

## Measured results for the bundled model

Evaluation freezes weights, uses 20 Hz controls and 120 Hz physics, and runs one nominal lap plus four recovery conditions on **each** track. Recovery starts use ±0.75 m offsets / ±0.07 rad heading at the grid and ±1 m / ±0.1 rad at 25% and 60% progress, with 6–8 m/s starting speed. A pass requires full-track completion, zero barrier contacts, and zero off-track time. Parked cars cannot pass regardless of lane score. Attempts have a 180-second limit and stop on prolonged off-track time, repeated contacts, or a stall.

| Nominal attempt | Neural time | Reference time | Score | Mean lane deviation | Off-track / contacts |
|---|---:|---:|---:|---:|---:|
| Alpine Circuit (train) | 51.9 s | 52.0 s | 95/100 | 0.164 m | 0 s / 0 |
| Canyon Run (held out) | 96.7 s | 91.6 s | 93/100 | 0.234 m | 0 s / 0 |

The bundled model passed **10/10** specified attempts. Canyon's nominal lateral trajectory MAE versus the reference is 0.065 m at matching track progress; mean action difference is 7.4% of normalized action range. The clean reference has small nonzero lane deviations; it is not literally flawless. Reference actions during evaluation are used only for scoring, never to control the learned car. Reports contain every attempt and progress trace.

Lane score: `clamp(100 - 32*meanAbsoluteDeviationMeters - 4*offroadSeconds - 8*barrierContacts, 0, 100)`. It covers the run since reset. Evaluation also compares completion, time, actions, and progress-aligned lateral deviation with a clean reference. These measurements establish generalization to the included circuits and stated recovery conditions. They do **not** establish flawless driving on arbitrary future tracks. Tighter curvature, different widths/dynamics, poor perception, or larger perturbations require new evaluation and possibly more training data. The bundled model was not tuned on Track 2 data.

## Reproduce offline training

Requires Python, NumPy, and PyTorch. TensorFlow.js is bundled for browser training; PyTorch trains the offline baseline.

```powershell
npm run data
npm run train
npm run evaluate
npm test
```

Data generation writes ignored `training/dataset.json`. Python trains with Adam for 100 epochs and exports portable weights to `dist/models/baseline.json`. Evaluation writes `dist/models/evaluation.json`. Initialization and dataset generation use seeds 2026; hardware/library differences can affect numerical reproducibility. Regenerate evaluation after replacing baseline weights.

## Files and checks

- `dist/physics.js`: dynamics and scripted reference controller.
- `dist/features.js`: action-free, road-relative observations.
- `dist/network.js`: model validation and pure neural forward inference.
- `dist/dataset.js`: demonstrations, augmentation, grouped splits, and manual-data ingestion.
- `dist/learning-worker.js`: browser training and frozen evaluation.
- `dist/evaluate.js`: closed-loop attempts and reference comparison.
- `dist/learning-ui.js`: learning curves, training, results, and model lifecycle.
- `scripts/`: reproducible data generation, PyTorch training, and evaluation.
- `tests/`: physics, steering taps, learned driving/recovery, frozen weights, zeroed-weights/no-fallback check, rotation/translation invariance, test-data exclusion, and model validation.

Keys: WASD / arrows, Space handbrake, R reset, C camera, Escape pause. Losing focus, opening a dialog, or starting a learning job pauses driving. Cameras, ideal line, reference ghost, road sensors, and fullscreen remain available.

Three.js 0.180.0 (MIT) and TensorFlow.js 4.22.0 (Apache-2.0) are vendored with licenses. Official TensorFlow.js docs: https://www.tensorflow.org/js/guide/train_models and https://www.tensorflow.org/js/guide/save_load.

## License

OpenNeural’s original code, documentation, and bundled model weights are available under the [MIT License](LICENSE). Vendored Three.js and TensorFlow.js retain their respective [MIT](dist/vendor/LICENSE) and [Apache 2.0](dist/vendor/TFJS-LICENSE) licenses.
