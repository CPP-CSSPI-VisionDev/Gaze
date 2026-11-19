# Switch 2 test notes

## Local demo on November 14 2026

I tested the gesture on my Mac and reported that these checks worked.
These are my manual test results, not a recorded accuracy benchmark.
We did not save frame-by-frame logs or a recording of the test runs.

| What I tried | Result I reported |
| --- | --- |
| 10 lower, raise, and hold cycles | Passed. One activation per cycle, ending ON after 10. |
| Hold the raised arm for 5 seconds | Passed. No extra activation while holding. |
| Raise briefly, for less than 1 second | Passed. No activation. |
| Move only the right arm | Passed. No activation. |
| Scratch my head, wave, and reach normally | Passed in my local test. No accidental activation. |
| Leave the frame and return with the left arm already raised | Passed. Had to lower it before activating again. |

## The tracking issue I found

The light stayed ON with activations at zero while the screen showed
LEFT ARM TRACKING UNCERTAIN. The cropped screenshot showed 0/3 usable
left-arm points and a minimum score of 0.01.

We also saw confidence values moving above and below 0.5. A short dip
used to reset the gesture straight away. The code now pauses for brief
dips, without counting that time as part of the hold. Longer uncertainty
still requires an arm-down reset.

The important camera finding was that the whole left arm needs to fit.
After I made the shoulder, elbow, and wrist visible, I confirmed the
gesture and simulated toggle worked. The confidence filter stayed at 0.5.

## Software checks during development

Artificial landmark checks passed for a one-second hold, no repeated
activation while holding, a second activation after lowering, brief
confidence dips, longer tracking loss, missing pose, and attention reset.
The model inference check also passed in the separate switch2 environment.
These checks do not replace camera testing or the real attention gate.

## Still needs testing

- The final elevated front-left camera view. This has not been confirmed.
- Ryan's actual gesture. The right-arm?
- Havish's real attention gate and losing attention during a hold.
- Physical switch control and whether the actual light state matches.
- I think that is all.


