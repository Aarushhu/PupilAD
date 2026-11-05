# PupilAD

PupilAD measures how much the pupil widens when a person holds information in working memory.
This is the task-evoked pupillary response. The code turns that response into per-person features.
The long-term aim is a low-cost marker of locus coeruleus function, which changes early in
Alzheimer's disease. **The current results do not support any diagnostic claim**

The repository has three parts:

1. **Analysis of a public dataset.** It cleans pupil recordings from OpenNeuro
   [ds003838](https://openneuro.org/datasets/ds003838) and tests whether pupil size tracks memory
   load.
2. **A task battery app.** It runs four memory and attention tasks on a laptop, phone or iPad while
   the camera records the participant's eyes, then extracts pupil size from the video.
3. **Support tools.** These read Tobii and pupillary light reflex exports, and include an LSTM model
   so far tested only on synthetic data.

## Task battery app

| Task                                            | Measures                           |
| ----------------------------------------------- | ---------------------------------- |
| Letter Match (2-back)                           | working memory updating            |
| Number Recall (digit span, forward and reverse) | working memory capacity            |
| Color Clash (Stroop)                            | attention and interference control |
| Animal Sprint (verbal fluency)                  | word retrieval, recorded as audio  |

The app checks camera framing and lighting before each session. Participants answer by tapping or
by using the keyboard, and each task has an optional practice round. The video is recorded in the
browser. After the session the app automatically saves the scores, a pupil-size trace and a
per-trial pupil table. Details, output files and timing accuracy are in [`docs/app.md`](docs/app.md).
