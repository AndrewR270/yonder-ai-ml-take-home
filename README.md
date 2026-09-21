# AI/ML Take Home: ML/AI Track

## Part 0: Base Knowledge Primer

### Object detection

Our rover needs to autonomously identify specific objects in its environment, most notably a mallet it has to locate and retrieve as part of our competition tasks. This happens onboard, in real time, using a live camera feed, under real-world lighting and framing conditions that are messier than a clean training set.

### Classification vs. detection

- **Classification:** "is this image a mallet or not" — one label per image.
- **Detection:** "where in this image is the mallet, if it's present at all" — a bounding box plus a label. This is the harder, more useful version, and what we actually run onboard.

You can approach this task as either, but detection is the closer match to our real use case and will be weighted accordingly if you attempt it (see rubric).

### Preprocessing

A common beginner mistake is assuming a bigger/fancier model fixes weak data. In practice, small, well-preprocessed datasets consistently outperform larger, messy ones for a narrow single-object task like this. Things worth considering: contrast/lighting normalization, background variety in your training set, augmentation (rotation, scale, brightness jitter) to compensate for a small dataset, and making sure your train/validation split doesn't leak near-duplicate frames (e.g., consecutive video frames of the same pose) between the two, which silently inflates your reported accuracy.

### Precision, recall, and accuracy

For a detection task like this, a model that simply never predicts "mallet" can still score deceptively well on naive accuracy if mallets are rare in your dataset. We care about **precision** (of the things you flagged as a mallet, how many actually were) and **recall** (of the actual mallets, how many did you catch). Report both, not just a single accuracy number.

### How Yonder actually uses this

Our object detection stack runs YOLO, migrated from a Jetson-based pipeline to run natively on an OrangePi's onboard NPU via RKNN, specifically to cut power draw and free up compute for navigation. Model size, inference speed, and real-world robustness (varying outdoor lighting, camera angle, partial occlusion) all matter as much as raw benchmark accuracy in our actual deployment. A model that scores well on a clean validation set but falls apart outdoors isn't useful to us.

### Resources

- [Ultralytics YOLO docs](https://docs.ultralytics.com/) (if using YOLO)
- [PyTorch tutorials](https://pytorch.org/tutorials/) (if building a classifier from scratch)
- [Roboflow](https://roboflow.com/) (useful for quick augmentation/preprocessing pipelines, optional)

## Starter Repo / Dataset

We provide:

- A labeled dataset of 1,000 images of a mallet and a bottle in varying backgrounds, lighting, and angles, in a standard format (YOLO-style bounding box annotations). Every image contains at least one labeled object, so you can also treat it as classification if you'd rather do that.
- A small held-out test set (not included in the training data) that we'll use to independently evaluate your final model. You won't have access to this set, so don't over-tune to your own validation split.

Your job is to build the actual training pipeline. This mirrors real ML work here: most of the effort should go into data handling, preprocessing, and evaluation judgment, not boilerplate.

### Getting the repo

You need [Git](https://git-scm.com/downloads). The repo is public, so no account or login is needed:

```bash
git clone https://gitlab.com/Yonder-Dynamics/take-home-projects/ai-ml-take-home.git
cd ai-ml-take-home
```

Don't want to use Git? Download [a ZIP of the repo](https://gitlab.com/Yonder-Dynamics/take-home-projects/ai-ml-take-home/-/archive/main/ai-ml-take-home-main.zip), unzip it, and work in the `ai-ml-take-home-main` folder. A ZIP has no Git history, so "Commit as you go" and "Submitting" below each have one extra step for you.

### What's in the repo

```
sampling/          the scripts we used to build the dataset (optional, see below)
requirements.txt   Python dependencies (keep it up to date as you add libraries)
AI_LOG.md          template for your AI usage log (Part 3)
METHODOLOGY.md     template for how to run your code and your thought process (Part 4)
```

### Getting the dataset

1. Go to [the dataset's download page](https://universe.roboflow.com/malletbottle2/sampled-yd-object-detection/dataset/1/download) and click **Download Dataset**. Roboflow may ask you to sign in or create a free account.
2. Choose the format and select the option to get a **code snippet or ZIP file**. YOLOv8 matches the YOLO-style labels described above; pick another format if your framework needs it.
3. Choose **Show download code**.
4. Wait for Roboflow to finish preparing (zipping) the files, then copy the code it shows, paste it into a Python file or notebook (`.ipynb`) in your repo, and run it.

The snippet looks like this:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

In a notebook, Roboflow's version starts with `!pip install roboflow`. In a `.py` file, run `pip install roboflow` in your terminal instead.

**Your API key is personal.** Roboflow puts it directly in the snippet. Don't commit it. Keep it in an environment variable or a git-ignored `.env` file, and remember your repo will be public.

Running it downloads the dataset into your repository, in a folder named `Sampled-YD-Object-Detection-1`. That folder is already in `.gitignore`, so the images won't be committed.

### What you get

```
Sampled-YD-Object-Detection-1/
  data.yaml
  train/images, train/labels     800 images
  valid/images, valid/labels     200 images
```

- All images are 512x512.
- There are two classes, listed in `data.yaml`: `bottle` (id 0) and `mallet` (id 1). Together the train and valid sets hold 702 bottle boxes and 537 mallet boxes. Report precision and recall for each class.
- Each image has a label file with one line per object: `class x_center y_center width height`, with the four numbers normalized between 0 and 1.
- There is no `test` folder (see the held-out set above). `data.yaml` still lists a `test` path, so remove that line if your framework complains.
- Many images are augmented variants (rotation, brightness and exposure changes) of photos from our larger dataset.
- License: CC BY 4.0 (also in `data.yaml`).

### The `sampling/` folder

This is how we built the dataset from our larger Roboflow project. You don't need it. If you'd like to draw a different sample, `sampling/sample-roboflow.py` does that (set `ROBOFLOW_API_KEY` in a `.env` file first; see `.env.example`), and `sampling/upload-roboflow.py` uploads a sample to your own Roboflow project. Its dependencies are in `requirements.txt`, which covers only these scripts, not model training.

## Part 1: Core Task (required)

1. Build a training pipeline (PyTorch, YOLO, or framework of your choice) using the provided dataset.
2. Implement at least one preprocessing/augmentation step beyond just resizing images, and briefly justify your choice(s) in your `METHODOLOGY.md` (e.g., why contrast normalization, why this augmentation set, given what you observed in the raw data).
3. Train your model and report precision, recall, and a confusion matrix or equivalent breakdown.
4. Submit your trained model weights (or a script to reproduce them) along with an inference script we can run against our held-out test set.
5. In your `METHODOLOGY.md`, include a short error analysis: look at a handful of your model's mistakes (false positives/negatives) and describe what you think is causing them.
6. Take a short video (your phone is fine) of a mallet-shaped object (or the closest household stand-in you have: a hammer, a rolling pin, whatever's on hand) in a real environment, and run your trained model against individual frames. Report how it performs outside the clean training distribution.

### Suggested order of work

Each step has a check that tells you it's right before you move on, and ends with a **commit** (see "Commit as you go" below). Stretch goals: one commit per goal, e.g. `stretch A: <name>`.

| Step | What to do | How you know it's done | Commit message |
| --- | --- | --- | --- |
| 1 | Get the repo (if you used the ZIP, run the `git init` lines below first) and download the dataset with a script that reads your API key from an environment variable. Look at a couple of dozen images and their labels. | `Sampled-YD-Object-Detection-1/` exists and does **not** show up in `git status`. Your key is nowhere in your files. | `step 1: dataset download script` |
| 2 | Explore the data: class balance, image sizes, lighting, where the objects sit in the frame. Note what you see in `METHODOLOGY.md`. | You can state three things about the raw data that will affect your choices. | `step 2: data exploration` |
| 3 | A baseline pipeline with only resizing: train, then evaluate on `valid/`. | It runs end to end and gives you precision and recall numbers to beat. | `step 3: baseline pipeline` |
| 4 | Add your preprocessing/augmentation step(s), retrain, and compare against the baseline. | Before/after numbers are written down, with why you chose it. | `step 4: preprocessing` |
| 5 | Evaluation: precision, recall and a confusion matrix (or equivalent) computed against the held-out split. | You can say what each number means for a mallet vs a bottle. | `step 5: evaluation` |
| 6 | Error analysis: look at a handful of false positives and false negatives and work out why. | Written in `METHODOLOGY.md` with specific examples. | `step 6: error analysis` |
| 7 | Inference script that runs on a folder of images, plus your weights (or a script that reproduces them). | It works from a fresh clone with one command. | `step 7: inference script` |
| 8 | Real-world test: your video, frames run through the model. | You can report how it does outside the clean training data. | `step 8: video test` |
| 9 | Pin `requirements.txt`, finish `METHODOLOGY.md` and `AI_LOG.md`, and test everything in a clean checkout. | Someone else could follow `METHODOLOGY.md` without asking you anything. | `step 9: methodology and requirements` |

### Commit as you go

We read your commit history as well as your code. It shows how you worked, and it is the honest record behind your write-up. Commit at the end of each step in the table above, using the message shown (or your own words in the same spirit).

One-time setup (git refuses to commit until it knows who you are). Use your own name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

At the end of each step:

```bash
git status                  # what changed? nothing surprising?
git add -A
git commit -m "step 2: <what you did>"
```

**If you downloaded the ZIP instead of cloning,** there is no history yet. Once you have set your name and email (above), and before step 1, run these, so that everything you change afterwards shows up as your own work:

```bash
git init -b main
git add -A
git commit -m "starter files (from ZIP)"
```

Rules of the road:

- **Small and honest beats tidy.** A history with `step 4: ...` followed by a `fix: ...` commit that repairs your own bug is exactly what we like to see. Fixing your own bug in a later commit is normal.
- **Do not squash, amend or force-push** to make the history look cleaner. Do not commit everything in one go at the end.
- **Commit your AI log as you go too.** When an AI tool gets something wrong, write the `AI_LOG.md` entry in the same commit as the fix.
- **Never commit your Roboflow API key**, the dataset folder, or your `.env`. Look at `git status` before every `git add -A`. If a key does get committed, do not try to rewrite history: revoke that key in Roboflow and make a new one.
- **Model weights:** GitHub rejects files over 100 MB. If your weights are large, commit the script that reproduces them instead.

### What we're looking for

- Does the pipeline actually run end-to-end and produce a working model?
- Is there a real preprocessing decision, not just default settings copy-pasted from a tutorial?
- Do they report and understand precision/recall separately, not just accuracy?
- Does the error analysis show genuine engagement with why the model fails where it fails, not just "accuracy was X%"?
- A completed `METHODOLOGY.md` that lets us run your code and explains your approach.

## Part 2: Stretch Goals (optional)

Pick any/all of these. Partial, well-reasoned attempts are valued over none.

- **Model efficiency:** Our real deployment target is an onboard NPU with real compute/power constraints, not a desktop GPU. Report your model's parameter count and estimated inference time, and briefly discuss what you'd trade off (accuracy vs. speed vs. size) if this had to run on constrained edge hardware. Attempting actual quantization or a smaller architecture variant is a bonus but not required.
- **Active learning / hard example mining:** Identify the training images your model is least confident about or gets wrong, and describe (or implement) a strategy for how you'd prioritize collecting more data to fix those specific failure modes, rather than just collecting more data blindly.

## Part 3: AI Usage Log (required)

Submit a short `AI_LOG.md` with your code (there is a template in the repo root). For each significant use of AI tools, note:

- What you asked
- What you kept vs. rewrote, and why
- Anything the AI got wrong that you had to catch

## Part 4: METHODOLOGY.md (required)

Edit the `METHODOLOGY.md` in the repo root (there is a template) so it covers:

- **How to run your code.** The exact steps for a reviewer to install everything and run your data download, training, evaluation and inference from a fresh clone and see it working. For example: one script that installs all the libraries, and one that runs everything and shows the output. Say which Python version and OS you tested on, and give the exact command to run your inference script on a folder of test images.
- **Your thought process, in bullet points.** Why you made the choices you did (data handling, preprocessing, model, evaluation), and what your results, error analysis and video test showed.

We read this alongside your code. Write it in your own words: we'd rather see clear reasoning and honest limitations than a polished description.

Keep your `requirements.txt` up to date. It must list every library your code needs, with versions pinned; the one in this repo only covers the sampling scripts. If your code downloads the dataset, read the Roboflow API key from an environment variable instead of writing it into the code, and name that variable in `METHODOLOGY.md` so we can set our own.

## Rubric

| Criterion | What we're scoring |
| --- | --- |
| Correctness | Pipeline runs end-to-end, produces a working, evaluable model |
| Preprocessing judgment | Real, justified preprocessing/augmentation decisions, not just tutorial defaults |
| Evaluation rigor | Reports precision/recall (not just accuracy), and the numbers are computed correctly against a proper held-out split |
| Error analysis | Genuine engagement with why the model fails on specific examples, not just a final metric |
| Handling ambiguity | How they resolved underspecified parts of the task: did they make a reasonable call and explain it? |
| AI verification | Evidence they tested/verified AI-assisted code and claims rather than taking them on faith (from log + code quality + commit history) |
| Stretch engagement (bonus, not required) | Attempted or completed any stretch goal |

We don't expect a perfect implementation. Those who show genuine effort and learning are the ones who will have a leg up!

---

## Submitting

1. **Create a public repository on your own GitHub account.**
2. **Point your clone at it.** Your clone's `origin` is our repo, which you can't push to:

   ```bash
   git remote set-url origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

   If you downloaded the ZIP instead of cloning, there is no `origin` yet. You already ran `git init` (see "Commit as you go"), so just add yours:

   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

3. **Check that it's public.** Open your repo's link in a private/incognito browser window. If you can see the code without logging in, so can we.
4. **Send us the link** in the Google Form you'll be asked to fill out.

Your repo should include your code, your trained model weights (or a script that reproduces them), your inference script, an up-to-date `requirements.txt`, your completed `METHODOLOGY.md`, your `AI_LOG.md`, and your **full commit history** (push all of it; do not squash). Don't commit your Roboflow API key or the downloaded dataset folder. `.env` is already git-ignored.
