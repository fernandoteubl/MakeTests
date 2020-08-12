# Manual

## Arguments
```
  -h, --help            show this help message and exit
  -v, --verbose         Increase output verbosity (most verbose: -vv).
  -s, --silent          Enable silent mode (disable almost all output).
  -d, --debug           Enable debug mode (show many windows and information).
  -q QUESTION_FILE, --question_file QUESTION_FILE
                        Run a specific question file for debug.
  -i, --interactive     Use interative console for debug.
  -t TEMPORARY_DIR, --temporary_dir TEMPORARY_DIR
                        Directory used by temporary dirs/files. If not used, a
                        temporary directory will be created and deleted in
                        sequence.
  -w {0,1,2,3,4,5,6,7,8,9}, --webcam {0,1,2,3,4,5,6,7,8,9}
                        Use webcam with ID.
  -p PDF, --pdf PDF     PDF file with all scanned tests to correct it.
  -e {choices,config,essay,number,ocr,questionanswer,truefalse}, --examples {choices,config,essay,number,ocr,questionanswer,truefalse}
```

## JSON Config

* **Comments**: Use "//" at first position of line.

* **Multiple lines**: "key": """First Line<breakline>Second Line""" is equal "key": ["First Line", "Second Line"]

* **includeJSON**:  Include a second JSON file to append or replace the current.
  * **overwite**: Just overwirte the key and value.
  * **adding**: If is a list, use "<key_name>+" to append or "+<key_name> to insert at begin.

* **questions**: A list with questions parameters.
  * **salt**: Used to use the same dataase and students, but other test. E.g.: Regular test and Sub test.
  * **db_path**: Path to question database.
  * **select**: A list of questions selected to include in current test from database.
    * **path**: Path to group of questions (subpath allowed) or a specific question.
    * **weight**: The weight of this question. It will be used in "final_calc" (we'll see later in this config)
    * **replaces**: A dict with variables to be replaced before to generate the test
  * **input**: List of studentes used to generate the tests
    * **filename**: Path to CSV file
    * **delimiter**: Delimiter used in CSV file
    * **quotechar**: Quotechar used in CSV file
  * **output**: PDFs generated
    * **tests**: Path to PDF with all tests
    * **template**: Path to template's PDF
  * **correction**: Information about correction of questions
    * **path**: Folder with all corrections information
    * **csv_file**: Path to CSV file with all students score
    * **delimiter**: Delimiter used in csv_file
    * **quotechar**: Quotechar used in csv_file
  * **headers**: The header of csv_file
    * **identification**: List of headers that identificate the student. This varialbes needs to exist in input students CSV file.
    * **counter**: Name of variable that has the number of question
    * **intermediate**: Intermediate headers with score of each question
    * **final**: Header with final score
  * **student_directory_id**: Name of directories for each student with specific feedback.
  * **final_calc**: A python function that take arguments list of score and weight of question and return the final score.
* **tex**: All LaTeX code used to generate the test.
  * **max_pages**: Mas number of pages for each test. It is necessary because each page has a unique QRCode, and the MakeTests needs to generate all images before of compile the LaTeX. A large number can slow the generating process.
  * **qrcode_id_must_be_concatenated_with_dash_plus_the_page_number**: The unique QRCode ID of student. It must be concatenated with "-<page_number>". In LaTeX preamble, use `\newcommand{\includeqrcodeimage}[2]{\includegraphics[width=2.4cm]{#1-#2.png}}` command, and call it in document by `\includeqrcodeimage{%QRCODE_ID%}{\thepage}`.
  * **question_image_answer_area**: All answer areas of each question is a image. This variable has the name this image.
  * **question_counter**: Variable with the number of current question (1, 2, 3, ..)
  * **question_total**: Variable with total number of questions.
  * **answer_text**: Used just in template's PDF, has the correct answer in text format.
  * **replaces**: A dict of variables to be replaced from document. This facilities the reuse of LaTeX template because you don't need to modify the values inside the latex code.
  * **includes**: All directories to include in LaTeX generation. You can add a path with all images, for example.
  * **preamble**: Preamble used in LaTeX.
  * **termination**: Final code in LaTeX.
  * **test**: Informations about individual test.
    * **header**: The LaTeX code used as header for each student.
    * **before**: The LaTeX code used before of each question.
    * **after**: The LaTeX code used after of each question.
    * **footer**: The LaTeX code used as footer for each student.
  * **template**: Informations about template.
    * **header**: The LaTeX code used in header in template.
    * **before**: The LaTeX code used before of student's answer
    * **answer**: The LaTeX code with the answer of each question from a specific student
    * **after**: The LaTeX code used after of student's answer
    * **footer**: The LaTeX code used as footer in template.

## List of Students

Student's list is a CSV file with delimiter and chotechar defined in JSON config. There is no hard header name, but all headers used needs to be configured in JSON config.


# First Test

## Create a Question's DataBase

Make directoreis and subdirectories

```
mkdir Questions
mkdir Questions/Easy
mkdir Questions/Medium
mkdir Questions/Hard
```

Create some questions from template

```
./MakeTests.py -e choices > Questions/Easy/choices.py
./MakeTests.py -e truefalse > Questions/Easy/truefalse.py
./MakeTests.py -e questionanswer > Questions/Medium/questionanswer.py
./MakeTests.py -e number > Questions/Medium/number.py
./MakeTests.py -e essay > Questions/Hard/essay.py
```

Edit/add/delete your questions.

## Create a config file

Create from template:

```
./MakeTests.py -e config > config.json
```

In question section, define the path to data base:

```
"db_path": "Questions"
```

Select number, groups and weights of questions:

```
"select": [
  {"path": "Easy",  "weight": 1, "replaces": {"%PREFIX%": "Weight 1"}},
  {"path": "Medium",  "weight": 2, "replaces": {"%PREFIX%": "Weight 2"}},
  {"path": "Medium",  "weight": 2, "replaces": {"%PREFIX%": "Weight 2"}},
  {"path": "Hard",  "weight": 1, "replaces": {"%PREFIX%": "Weight 2"}}
]
```

In tex/includes section, insert all additional paths:

```
  "includes": [
    "img"
  ],
```

The template has a logo called logo.jpeg.
Or remove it from template, or add a logo.jpeg in img path.

Create a CSV student's list file called `Students.csv`. Example:

```
%ID%;%NAME%;%EMAIL%
000001;"Alice";alice@maketests.com
000002;"Bob";bob@maketests.com
```

# Errors and solutions

**Error:** LaTeX error, no idea.

**Solution:**
1. Create a folder called tex;
1. Include -t tex as parameter;
1. Run `pdftex` manually and see the tex source.

**Error:** There is a %SOMETHING% in LaTeX.

**Solution:** Check the Students.csv header. Some variable is wrong and wasn't replaced.
