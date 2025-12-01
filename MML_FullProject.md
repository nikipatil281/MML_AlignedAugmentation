🧠 **Full Project Explanation in Simple English**
=================================================

_From baseline → semantic pairing → mixing → alignment-aware augmentation_
--------------------------------------------------------------------------

🔵 1. **What problem are we solving?**
======================================

You want a system that can:

*   Given an **image**, retrieve its **caption**
    
*   Given a **text caption**, retrieve its **image**
    

This is called **image–text retrieval**.

You use the **MSCOCO dataset**, which contains:

*   ~120,000 images
    
*   Each image has **5 different human-written captions**
    
*   Each image also has **object labels** (like “dog”, “bed”, “person”, “bus”)
    

Your project builds a **multimodal embedding model** that puts images and captions together in the same vector space.

🟢 2. **The baseline system (first part of the project)**
=========================================================

### What it does:

*   Takes an image → produces a vector (size 256)
    
*   Takes a caption → produces a vector (size 256)
    
*   Trains the model so that **matching image–caption pairs end up close together** and **non-matching pairs end up far apart**.
    

### Python files involved:

### **(A) Image encoder (ResNet50)**

File: src/models/retrieval\_model.py

*   Takes a **224×224 RGB image**
    
*   Feeds it through a **pretrained ResNet**
    
*   Removes the classification layer
    
*   Outputs a **2048-dimensional feature**
    
*   Then compresses it down to **256 dimensions**
    
*   This becomes the “image embedding”
    

Example:

> An image of a dog sleeping becomes a vector like \[0.10, -0.08, 0.67, ...\] of length 256.

### **(B) Text encoder (BERT)**

Same file.

*   Takes a caption like:_“A dog laying on a bed with a TV in the background.”_
    
*   Runs it through BERT
    
*   Takes the “CLS” output
    
*   Converts that into a **256-dimensional embedding**
    

Example:

> The caption becomes another 256-dimensional vector.

### **(C) Retrieval training script**

File: scripts/train\_retrieval.py

This script:

*   Loads COCO images + captions
    
*   Puts them into batches
    
*   Runs them through the ResNet50 + BERT encoders
    
*   Teaches the model so that:
    
    *   Correct pairs are closer
        
    *   Wrong pairs are farther
        

This gives you a **baseline**.

### Why we started here

To make sure the dataset, loaders, ResNet, BERT, and retrieval code all work.

### **(D) Evaluation script**

File: scripts/eval\_retrieval.py

*   Encodes **all images** and **all captions**
    
*   Computes similarity between every image and every caption
    
*   Reports top-1, top-5, and top-10 retrieval accuracy
    

This becomes your **baseline performance**, e.g.:

*   R@1 ≈ 22%
    
*   R@5 ≈ 49%
    
*   R@10 ≈ 63%
    

🟡 3. **What we change next: we add “synthetic training examples”**
===================================================================

The baseline learns only from **actual COCO pairs**.

But modern multimodal systems improve when you add **extra training examples** created by:

*   Mixing features
    
*   Perturbing representations
    
*   Blending semantics
    
*   Combining different image/text signals
    

This is called **data augmentation in feature space**, and this is where your novel work begins.

🟠 4. **LeMDA-Lite (your first novel component)**
=================================================

### What this does:

It adds a **learnable network** that automatically creates **“harder” versions** of your image–text pairs during training.

Think of it as:

> Taking the original representation and nudging it in a way that forces the main model to become more robust.

### Python file:

src/models/augmentor.pyand training scriptscripts/train\_retrieval\_lemdalike.py

### In simple English:

*   The image encoder produces a 256-dim vector.
    
*   The text encoder produces a 256-dim vector.
    
*   We concatenate them (512-dim).
    
*   The **augmentor network** slightly **modifies** this vector.
    
*   The model must still understand this modified version correctly.
    

This gives your model **extra synthetic training pairs**.

### Why this is useful:

You get more diverse training examples without needing more data.

🟣 5. **Learnable Semantic Pairing**
====================================

(_Your second novel contribution_)

### Problem we solved:

LeMDA mixes or perturbs randomly.Random mixing **can create garbage samples**, hurting alignment.

You solved this by letting the model **learn which samples are semantically related**.

### Python file:

src/models/pairing.pyandscripts/train\_retrieval\_lemdalike\_pairing.py

### What it does in simple English:

*   Every sample (image + caption pair) gets turned into a special “pairing vector”
    
*   Samples with **the same objects** (dogs, beds, cars, etc.) should be close
    
*   Samples with **different objects** should be far
    
*   The model learns this using the COCO object labels
    

So instead of:

> “Mix sample #84 with #195 randomly”

We can now say:

> “Mix this dog-on-a-bed sample with another dog-on-a-bed or similar bedroom scene.”

### Why this is good:

Mixing is now **meaningful**, not random.

🔴 6. **Global multimodal mixing ("Hybrid Premix")**
====================================================

(_Your third novel component_)

Next, we actually **use** the learned pairing network to select meaningful partners for mixing.

### Python file:

scripts/train\_retrieval\_hybrid\_premix.py

### What happens:

*   For a given image–caption pair,the pairing network finds the **most semantically similar** other pair in the batch.
    
*   The system mixes:
    
    *   Image features with weight λ
        
    *   Text features with weight 1−λ
        
*   These λ values depend on how similar the pairs are.
    

This creates **semantically consistent synthetic examples**:

*   Mixing two bedroom scenes
    
*   Mixing two outdoor sports scenes
    
*   Mixing two images involving people holding objects
    

### Why this matters:

Before, augmentations were “blind”.Now they’re **aware of semantic structure**.

This is your **core hybrid augmentation idea**.

🟤 7. **TokenMix (patch-level + token-level mixing)**
=====================================================

(_Your fourth novel contribution_)

So far, mixing was done at the **global** 256-dim vector level.But global mixing is coarse.

Now we go **deeper**:

### We mix **visual patches** and **text tokens individually**.

### Python file:

src/models/retrieval\_tokens\_model.pyandscripts/train\_retrieval\_hybrid\_tokenmix.py

### What happens:

#### For images:

*   ResNet feature maps give you a **7×7 grid of patches**
    
*   Each patch becomes a 256-dim token
    
*   Now you mix these patch tokens between two images
    

Example:Mixing dog-sleeping image with dog-sitting image gives a blended patch-level representation:

*   Some bed regions from image A
    
*   Some dog body regions from image B
    

#### For text:

*   BERT gives token embeddings for each word
    
*   We blend individual word features
    
*   Example:“dog” + “puppy” → combined dog-related token
    

### Why this is powerful:

You are now mixing **semantically aligned subparts**, not whole images.This is much more fine-grained and creates **richer synthetic examples**.

⚫ 8. **Alignment-Aware Augmentor**
==================================

(_Your fifth and highest-level novelty_)

Now that we mix patches/tokens, we want the augmentor to know:

*   How well the anchor image and caption matched originally
    
*   How well the mixed version matches
    
*   How many object labels are in the scene
    
*   Whether the mixed result is confusing or coherent
    

So we extend the augmentor:

### Python file:

src/models/align\_augmentor.pyandscripts/train\_tokenmix\_align\_full\_coco.py

### What it does:

We feed into the augmentor:

1.  **Image–caption similarity BEFORE mixing**
    
2.  **Image–caption similarity AFTER mixing**
    
3.  **Label density** (how many objects are in this image)
    

This gives the augmentor **context** so it can:

*   Push representations that make the retrieval model more robust
    
*   Avoid too-destructive augmentations
    
*   Focus on meaningful modifications
    

This is your **final, novel architecture**:

> **Learned semantic pairing + structured patch/token mixing + alignment-aware augmentation.**

🎉 9. **What you created in the end**
=====================================

Your full system now:

1.  Takes a real COCO batch
    
2.  Learns which samples are semantically related
    
3.  Mixes corresponding image patches and word tokens
    
4.  Computes alignment information
    
5.  Feeds the mixed features into an augmentor that modifies them intelligently
    
6.  Trains the retrieval model to handle both real and augmented samples
    

You have rebuilt a **multimodal, alignment-aware, semantically structured augmentation framework**.

This is _not_ a small project — this is graduate-level research.

📁 10. **Summary of what each Python file does (simple English)**
=================================================================

### **src/models/retrieval\_model.py**

*   Takes in a 224×224 image
    
*   Uses ResNet to turn it into a 256-dim vector
    
*   Takes in a caption
    
*   Uses BERT to turn it into a 256-dim vector
    
*   Used for baseline retrieval
    

### **src/models/retrieval\_tokens\_model.py**

*   Extracts 7×7 visual patch tokens
    
*   Extracts BERT token embeddings
    
*   Produces both patch-level and global embeddings
    
*   Used for TokenMix & alignment-aware model
    

### **src/models/augmentor.py**

*   Takes a fused image–text vector (size 512)
    
*   Slightly perturbs it
    
*   Used to generate harder training examples
    

### **src/models/align\_augmentor.py**

*   Takes:
    
    *   mixed image–text vector
        
    *   BEFORE/AFTER alignment metrics
        
    *   label density
        
*   Outputs a smarter perturbation
    
*   Used only in final model
    

### **src/models/pairing.py**

*   Learns which pairs of images should be mixed
    
*   Uses object labels to understand similarity
    
*   Outputs a semantic “pairing” vector
    

### **scripts/train\_retrieval.py**

*   Trains the baseline model
    
*   Only real images + captions
    
*   No synthetic mixing
    

### **scripts/train\_retrieval\_lemdalike.py**

*   Adds augmentor network
    
*   Creates harder variants of samples
    
*   Model trains on both original + perturbed samples
    

### **scripts/train\_retrieval\_lemdalike\_pairing.py**

*   Adds pairing network
    
*   Learns semantic similarity of samples
    
*   Still no mixing yet
    

### **scripts/train\_retrieval\_hybrid\_premix.py**

*   Uses pairing network to pick who to mix with
    
*   Blends image and text features globally
    
*   Produces meaningful synthetic examples
    

### **scripts/train\_retrieval\_hybrid\_tokenmix.py**

*   Mixes **individual patches** and **individual text tokens**
    
*   Provides fine-grained synthetic examples
    

### **scripts/train\_tokenmix\_align\_full\_coco.py**

*   Adds alignment-aware augmentor
    
*   Trains full system on full COCO
    
*   This is your final model
    

### **scripts/eval\_retrieval\_full\_coco.py** / eval\_tokenmix\_align\_full\_coco.py

*   Tests the models on COCO val2017
    
*   Reports retrieval accuracy
    
*   Used for comparing baseline vs. your method
    

### **scripts/vis\_pairing\_neighbors.py**

*   Shows images the pairing network thinks are similar
    
*   Demonstrates semantic learning
    

### **scripts/vis\_patch\_similarity.py**

*   Shows patch similarity maps between anchor + partner
    
*   Demonstrates patch-level structure learned by TokenMix
    

### **scripts/vis\_autoencoder\_mixed\_images.py**

*   Optional “toy” experiment
    
*   Uses a small autoencoder to actually decode mixed latents
    
*   Produces real synthetic mixed images
    
*   Just for intuition (not part of the main model)
    

💎 11. **Your Novelty (summarized)**
====================================

You introduced **three major innovations**:

### **(1) Learned Semantic Pairing**

*   Instead of random mixing, the model learns _which samples are compatible_ for mixing.
    
*   Uses COCO category labels to supervise this.
    

### **(2) Structured Multimodal Mixing (TokenMix)**

*   Mixing is done:
    
    *   Per visual patch
        
    *   Per text token
        
*   Mixing strength depends on semantic similarity.
    

This is far more detailed than standard MixGen or simple vector mixing.

### **(3) Alignment-Aware Augmentation**

*   Augmentor receives:
    
    *   Original similarity
        
    *   Mixed similarity
        
    *   Label density
        
*   Produces context-aware perturbations.
    

This makes the augmentation _adaptive_ and _alignment-sensitive_.

⭐ 12. In one sentence, what you built
=====================================

> **A hybrid multimodal data augmentation system that uses learned semantic pairing, fine-grained patch/token mixing, and alignment-aware modifications to enrich training for image–text retrieval on MSCOCO.**