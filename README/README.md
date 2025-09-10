# AI-Powered Autocorrect Tool

A sophisticated web-based autocorrect tool that corrects spelling errors using statistical NLP techniques and Levenshtein distance algorithm.

## Features

- Real-time spelling correction using statistical word frequency data
- Clean, responsive web interface built with Tailwind CSS
- Advanced algorithm using Levenshtein distance and word frequency ranking
- Data visualization of top word frequencies
- Error handling for various input scenarios

## Algorithm Overview

The autocorrect system works through these steps:

1. **Text Preprocessing**: Input text is tokenized and normalized using NLTK
2. **Candidate Generation**: For each potentially misspelled word, generates correction candidates with small edit distances (1-2)
3. **Candidate Ranking**: Candidates are ranked by edit distance (primary) and word frequency (secondary)
4. **Correction Selection**: The highest-ranked candidate is selected as the correction

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or download this project
2. Navigate to the project directory
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
