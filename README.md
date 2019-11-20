# Analysis-of-EEG-Recordings-from-Music-Perception-and-Imagination

> Our purpose is to use music imagery information retrieval systems to recognize a song from only subjects' thoughts by electroencephalography (EEG) recordings taken during music perception and imagination. Music imagination can be either imagining themselves producing the music or simply hearing the music in his head. OpenMIIR dataset is released from Brain and Mind Institute, Department of Psychology, Western University, London. There are nine subjects listening to twelve songs respectively. The twelve songs belong to three categories - with lyrics, without lyrics, and instrumental pieces. The lengths of songs vary from 6.9 seconds to 16.0 seconds.

## Proposed Method
> Pre-processing
  - Remove noise
    1. Remove bad channels
    2. Bandpass filtering (0.5 - 30 Hz)
    3. Down-sampling to 256 Hz
    4. Remove artifacts caused by eye blinks using extended Infomax ICA
  - Data Segment
    Split all data into 3, 4, 5, 6 seconds individually 
    (1) first segment
    (2) as many as possible
     - using slide window = 0.5 seconds

> Deep Neural Model
  - Method 1: 1 Stage
    - 12 class:
    Directly use 12 songs to train the model
    - Use 10-fold validation
  - Method 2: 2 Stage
    - 1st layer: 3 class
    Split data into three categories (with lyrics / without lyrics / instrumental pieces)
    - 2nd layer: 4 class
    Each category trains a model
    - Use 10-fold validation

## Conclusion
> The 1-stage model classifies the dataset to each music and gets the approximate same accuracy as the paper. Thus, we evaluate the model to fit each category and find that there is over fifty percent to classify correctly.
> The two-stage model pre-trains the dataset to each category firstly then classifies them to each music. It gains 3%~4% accuracy than the 1-stage model. 
> Also, the model for single-person is much better than the common version due to the coherence of datum. And because of the data quantity is proportional to the differences of each music, the accuracy increases progressively with seconds. Overall, our results gain 2%~6% accuracy than the paper.
