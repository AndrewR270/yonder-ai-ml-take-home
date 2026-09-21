# AI/ML Take Home: ML/AI Track

## Part 0: Base Knowledge Primer

### Object detection

Our rover needs to autonomously identify specific objects in its environment, most notably a mallet it has to locate and retrieve as part of our competition tasks. This happens onboard, in real time, using a live camera feed, under real-world lighting and framing conditions that are messier than a clean training set.

### Classification vs. detection

- **Classification:** "is this image a mallet or not" (one label per image).
- **Detection:** "where in this image is the mallet, if it's present at all" (a bounding box plus a label). This is the harder, more useful version, and what we actually run onboard.

You can approach this task as either, but detection is the closer match to our real use case and will be weighted accordingly if you attempt it (see rubric).

### Preprocessing

A common beginner mistake is assuming a bigger/fancier model fixes weak data. In practice, small, well-preprocessed datasets consistently outperform larger, messy ones for a narrow single-object task like this. Preprocessing is particularly good at increasing performance (things like contrast/lighting normalization, augmentation, etc.).

### Evaluation

Detection doesn't have a single "accuracy" number, and one score can hide a lot. The two classes here are fairly balanced (685 bottle boxes, 537 mallet boxes), so imbalance isn't the main risk. Think about which numbers tell you whether the model works, and report them for each class.

There's good reason for this. Imagine we had a dataset of which 90% were images of mallets and 10% were images of bottles. If we wrote a naive model to always predict that we found a mallet, we would have an accuracy of 90%, but that hides the fact that our model is never able to find a bottle!

### How Yonder actually uses this

Our object detection stack runs YOLO, migrated from a Jetson-based pipeline to run natively on an OrangePi's onboard NPU via RKNN, specifically to cut power draw and free up compute for navigation. Model size, inference speed, and real-world robustness (varying outdoor lighting, camera angle, partial occlusion) all matter as much as raw benchmark accuracy in our actual deployment. A model that scores well on a clean validation set but falls apart outdoors isn't useful to us.

### Resources

- [Ultralytics YOLO docs](https://docs.ultralytics.com/) (if using YOLO)
- [PyTorch tutorials](https://pytorch.org/tutorials/) (if building a classifier from scratch)
- [Roboflow](https://roboflow.com/) (useful for quick augmentation/preprocessing pipelines, optional)

## Starter Repo / Dataset

We provide:

- A labeled dataset of 999 images of a mallet and a bottle in varying backgrounds, lighting, and angles, in a standard format (YOLO-style bounding box annotations). Every image contains at least one labeled object, so you can also treat it as classification if you'd rather do that.

Your job is to build the training pipeline. Most of the effort should go into the data, the preprocessing and the evaluation, not boilerplate, as it would on a real ML project.

### Getting the repo

You need [Git](https://git-scm.com/downloads). The repo is public, so no account or login is needed:

```bash
git clone https://gitlab.com/Yonder-Dynamics/take-home-projects/ai-ml-take-home.git
cd ai-ml-take-home
```

### What's in the repo

```
sampling/          the scripts we used to build the dataset (optional, see below)
requirements.txt   Python dependencies (keep it up to date as you add libraries)
AI_LOG.md          template for your AI usage log (Part 3)
METHODOLOGY.md     template for how to run your code and your thought process (Part 4)
```

### Files you shouldn't edit

- `sampling/`. It's how we built the dataset, not part of your solution.
- The downloaded dataset folder (`Sampled-YD-Object-Detection-2/`). Treat it as read-only. If you preprocess or relabel anything, write the result to a new folder, so anyone can delete the download, fetch it again, and still reproduce your results.

Everything else is yours, including `requirements.txt`, which you should update as you add libraries.

### Setting up Python

You need Python 3.12 or newer (we tested with 3.13). From the repo folder, make a virtual environment so your libraries stay separate from the rest of your computer:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` only covers the sampling scripts. Install whatever you train with (for example `pip install ultralytics`), then add it to `requirements.txt` with its version pinned (`pip freeze` shows what you have installed).

Training is much faster on a GPU. In our tests, training YOLOv8n on the training images at 512x512 took about 2.5 minutes per epoch on a fast 8-core laptop CPU (roughly 100 minutes for 40 epochs) and about 6 seconds per epoch on an NVIDIA RTX 4070 laptop GPU (roughly 6 minutes for 40 epochs). You don't need a GPU, but on a CPU plan your time accordingly, or use a free GPU on [Google Colab](https://colab.research.google.com) (Runtime > Change runtime type > GPU).

On Windows, plain `pip install ultralytics` installs a CPU-only PyTorch. If you have an NVIDIA GPU, install the CUDA build of PyTorch first (use the selector on [pytorch.org](https://pytorch.org/get-started/locally/)), then check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

If you train from a `.py` file on Windows, put the training code in a function and call it under `if __name__ == "__main__":`. Without that, the data loader workers crash at startup with an error about starting a new process before the current one has finished bootstrapping.

### Getting the dataset

1. Go to [the dataset's download page](https://universe.roboflow.com/malletbottle2/sampled-yd-object-detection/dataset/2/download) and click **Download Dataset**. Roboflow may ask you to sign in or create a free account.
2. Choose the format and select the option to get a **code snippet or ZIP file**. YOLOv8 matches the YOLO-style labels described above; pick another format if your framework needs it.
3. Choose **Show download code**.
4. Wait for Roboflow to finish preparing (zipping) the files, then copy the code it shows, paste it into a Python file or notebook (`.ipynb`) in your repo, and run it.

The snippet looks like this:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("malletbottle2").project("sampled-yd-object-detection")
version = project.version(2)
dataset = version.download("yolov8")
```

In a notebook, Roboflow's version starts with `!pip install roboflow`. In a `.py` file, run `pip install roboflow` in your terminal instead.

**Your API key is personal.** Roboflow puts it directly in the snippet. Don't commit it. Keep it in an environment variable or a git-ignored `.env` file, and remember your repo will be public. Error messages from the Roboflow library can include your key in a URL, so remove it before pasting an error into a chat, an issue or an AI tool.

Running it downloads the dataset into your repository, in a folder named `Sampled-YD-Object-Detection-2`. That folder is already in `.gitignore`, so the images won't be committed.

**If the download fails with `SSLCertVerificationError`:** some antivirus programs and campus or corporate networks re-sign HTTPS traffic with a certificate that Python doesn't trust by default. Run `pip install truststore` and add these two lines at the top of your download script, so Python uses your operating system's certificates:

```python
import truststore
truststore.inject_into_ssl()
```

Don't turn certificate checking off.

### What you get

```
Sampled-YD-Object-Detection-2/
  data.yaml
  train/images, train/labels     800 images
  valid/images, valid/labels     199 images
```

- There are two classes, listed in `data.yaml`: `bottle` (id 0) and `mallet` (id 1). Report your results for each class.
- Each image has a label file with one line per object: `class x_center y_center width height`, with the four numbers normalized between 0 and 1.
- There is no `test` folder. `data.yaml` still lists a `test` path, so remove that line if your framework complains.
- Many images are augmented variants (rotation, brightness and exposure changes) of photos from our larger dataset.
- License: CC BY 4.0 (also in `data.yaml`).

### The `sampling/` folder

This is how we built the dataset from our larger Roboflow project. You don't need it. If you'd like to draw a different sample, `sampling/sample-roboflow.py` does that (set `ROBOFLOW_API_KEY` in a `.env` file first; see `.env.example`), and `sampling/upload-roboflow.py` uploads a sample to your own Roboflow project. Its dependencies are in `requirements.txt`, which covers only these scripts, not model training.

## Part 1: Core Task (required)

1. Build a training pipeline (PyTorch, YOLO, or framework of your choice) using the provided dataset.
2. Implement at least one preprocessing/augmentation step beyond just resizing images, and briefly justify your choice(s) in your `METHODOLOGY.md` (e.g., why contrast normalization, why this augmentation set, given what you observed in the raw data).
3. Train your model and report relevant metrics.
4. Submit your trained model weights (or a script to reproduce them) and an inference script we can run on a folder of images. It should take the image folder and an output folder as arguments. For each image it writes a `.txt` file with the same name, one detection per line: `class_id x_center y_center width height confidence`. Use the same normalized 0-1 coordinates and class ids as the dataset labels, and say which confidence threshold you used.
5. In your `METHODOLOGY.md`, include a short error analysis: look at a handful of your model's mistakes (false positives/negatives) and describe what you think is causing them.
6. Take a short video (your phone is fine) of a mallet-shaped object (or the closest household stand-in you have: a hammer, a rolling pin, whatever's on hand) in a real environment, and run your trained model against individual frames. Report how it performs outside the clean training distribution.

### Commit as you go

We read your commit history as well as your code. It shows how you worked, and it backs up your `METHODOLOGY.md`. Commit as you go, in small steps.

One-time setup (git refuses to commit until it knows who you are). Use your own name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Each time you finish something:

```bash
git status                  # what changed? nothing surprising?
git add -A
git commit -m "<what you did>"
```

Other commit rules:

- Don't worry about a tidy history. Fixing your own bug in a later commit is normal.
- Do not squash, amend or force-push to make the history look cleaner, and do not commit everything in one go at the end.
- **Never commit your Roboflow API key**, the dataset folder, or your `.env`. Look at `git status` before every `git add -A`. If a key does get committed, do not try to rewrite history: revoke that key in Roboflow and make a new one.
- GitHub rejects files over 100 MB. If your weights are large, commit the script that reproduces them instead.

### What we're looking for

- Does the pipeline actually run end to end and produce a working model?
- Is there a real preprocessing decision, not just default settings copy-pasted from a tutorial?
- Do they choose metrics that suit the task and understand what they mean, rather than defaulting to accuracy?
- Does the error analysis explain why the model fails where it does, rather than just reporting "accuracy was X%"?
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
- **Your thought process, in bullet points.** Why you made the choices you did (data handling including how you split train and validation, preprocessing, model, evaluation), and what your results, error analysis and video test showed.

We read this alongside your code. Write it in your own words: we'd rather see clear reasoning and honest limitations than a polished description.

Keep your `requirements.txt` up to date. It must list every library your code needs, with versions pinned; the one in this repo only covers the sampling scripts. If your code downloads the dataset, read the Roboflow API key from an environment variable instead of writing it into the code, and name that variable in `METHODOLOGY.md` so we can set our own.

## Rubric

| Criterion | What we're scoring |
| --- | --- |
| Correctness | Pipeline runs end to end and produces a working model, and the results you report hold up |
| Legibility | Results and error analysis are easy to read and check |
| Design judgment | Evidence of intentional choices beyond the minimum ask (preprocessing, how you split the data, model and settings) |
| Handling ambiguity | How did they resolve underspecified parts of the task? Did they make a reasonable call and explain it? |
| Understanding, not just output | Can they explain why the model works and where it fails? Does `METHODOLOGY.md` show real comprehension? |
| AI verification | Evidence they tested/verified AI-assisted code and claims rather than taking them on faith (from log + code quality) |
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

3. **Check that it's public.** Open your repo's link in a private/incognito browser window. If you can see the code without logging in, so can we.
4. **Send us the link** in the Google Form you'll be asked to fill out.

Your repo should include your code, your trained model weights (or a script that reproduces them), your inference script, an up-to-date `requirements.txt`, your completed `METHODOLOGY.md`, your `AI_LOG.md`, and your **full commit history** (push all of it; do not squash). Don't commit your Roboflow API key or the downloaded dataset folder. `.env` is already git-ignored.
