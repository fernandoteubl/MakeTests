# What is it?
* See the [demo video](https://drive.google.com/file/d/1npVosL80B5xCOx8zNdWy3lCc8zuD4b2e/view?usp=sharing)

# Manual

## Install

* MACPORT
  * Install [MacPorts](https://www.macports.org)
    * Execute in Terminal:
    * `sudo port -v selfupdate`
* ZBAR
  * MAC OSX:
    * `xcode-select --install`
    * `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
    * `brew install zbar`
  * LINUX:
    * `sudo apt-get install libzbar-dev libzbar0`

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

* **Multiple lines**: "key": """First Line\<breakline\>Second Line""" is equal "key": ["First Line", "Second Line"]

* **includeJSON**:  Include a second JSON file to append or replace the current.
  * **overwite**: Just overwirte the key and value.
  * **adding**: If is a list, use "<key_name>+" to append or "+<key_name> to insert at begin.

* **questions**: A list with questions parameters.
  * **salt**: The salt used in random functions to allow use the same dataase and students list, but generate a different random test. E.g.: Regular test and Sub test.
  * **db_path**: Path to question database.
  * **select**: A list of questions selected to include in current test from database.
    * **path**: Path to group of questions (subpath allowed) or a specific question.
    * **weight**: The weight of this question. It will be used in "final_calc" (we'll see later in this config).
    * **replaces**: A dict with variables to be replaced in LaTeX Question before.
  * **input**: The list of studentes used to generate the tests.
    * **filename**: Path to CSV file.
    * **delimiter**: Delimiter used in CSV file.
    * **quotechar**: Quotechar used in CSV file.
  * **output**: PDFs generated.
    * **tests**: Path to PDF with all tests.
    * **template**: Path to template's PDF.
  * **correction**: Information about correction of questions.
    * **path**: Folder with all corrections information.
    * **csv_file**: Name of CSV file with all students score.
    * **delimiter**: Delimiter used in csv_file.
    * **quotechar**: Quotechar used in csv_file.
  * **headers**: Headers of csv_file.
    * **identification**: A dict with headers used to identificate the student. The key is the header used in students.csv and the value is the alias used in header of students score CSV file.
    * **counter**: Name of variable that has an integer number of the current question.
    * **intermediate**: Template used in intermediate headers with score of each question.
    * **final**: Header with final score.
  * **student_directory_id**: Name of directories for each student with specific feedback.
  * **final_calc**: A python function that take arguments list of score and weight of question and return the final score.
* **tex**: All LaTeX code used to generate the test.
  * **max_pages**: Mas number of pages for each test. It is necessary because each page has a unique QRCode image, and the MakeTests needs to generate all images before of compile the LaTeX. A large number of QRCode images can slow the generating process.
  * **qrcode_id_must_be_concatenated_with_dash_plus_the_page_number**: The unique QRCode ID of student-page. It must be concatenated with "-<page_number>". In LaTeX preamble, use `\newcommand{\includeqrcodeimage}[2]{\includegraphics[width=2.4cm]{#1-#2.png}}` command, and call it in document by `\includeqrcodeimage{%QRCODE_ID%}{\thepage}`.
  * **question_image_answer_area**: All answer areas of each question is a image. This variable has the name of this image.
  * **question_counter**: Variable with the number of current question (1, 2, 3, ..).
  * **question_total**: Variable with total number of questions.
  * **answer_text**: Used just in template's PDF, has the correct answer in text format.
  * **replaces**: A dict of variables to be replaced from document. This facilities the reuse of LaTeX template because you don't need to modify the values inside of LaTeX code.
  * **includes**: All directories to include in LaTeX generation. E.g.: "img", and include all images inside of this directory.
  * **preamble**: Preamble used in LaTeX.
  * **termination**: Termination used in LaTeX.
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

Student's list is a CSV file with delimiter and chotechar defined in JSON config.
There is no hard header name, but all headers used needs to be configured in JSON config.

# Example of use

## Create a Question's DataBase

Create directories and subdirectories of database:

```
mkdir Questions
mkdir Questions/Easy
mkdir Questions/Medium
mkdir Questions/Hard
```

Create some questions from template:

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

Edit `config.json` if necessary.

The template has a logo called `logo.jpeg`.
You can remove it from LaTeX template or add a logo.jpeg in img path:

```
mkdir img
curl -0 https://github.githubassets.com/images/modules/logos_page/Octocat.png -o img/logo.png
```

Create a CSV student's list file called `Students.csv`.

To genarate a random `Students.csv` file, run this script:

```
echo "%ID%;%NAME%;%EMAIL%" > Students.csv
for i in {1..30} ; do
	nw=$[RANDOM%3+2]
	name=""
	for (( i = 1; i <= $nw; i++ )); do
		set="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		name+=${set:$RANDOM % ${#set}:1}
		set="abcdefghijklmonpqrstuvwxyz"
		n=$[RANDOM%12+4]
		for j in `seq 1 $n`; do; char=${set:$RANDOM % ${#set}:1}; name+=$char; done
		if [ $i -lt $nw ]; then; name+=" "; fi
	done
	mail=""
	set="abcdefghijklmonpqrstuvwxyz"
	n=$[RANDOM%12+4]
	for j in `seq 1 $n`; do; char=${set:$RANDOM % ${#set}:1}; mail+=$char; done
	mail+="@"
	for j in `seq 1 $n`; do; char=${set:$RANDOM % ${#set}:1}; mail+=$char; done
	mail+=".com"
	echo "$[RANDOM%100000000+100000000];\"$name\";$mail" >> Students.csv
done
```

## Generate PDFs

Just run:

`MakeTests.py`

You can add `-v` or `-vv` to most verbose.

## Print and Apply

Print the `Tests.pdf` using duplex option (double side).

The `Template.pdf` is optional.

## Correcting

### Using WebCAM

Just run:

`MakeTesta.py -w 0`

*NOTE: Try -w 1, -w 2, ... to select the correct WebCam*

Point the camera at the questions!

### Using PDF Scanned

Scans all tests in a single PDF.

```
./Maketests.py -p <name_of_scanned_tests_file.pdf>
```

If you have more than one PDF, execute multiple times to append all scores and feedbacks.

Some PDFs format is incompatible with MakeTests.
If an error occurs, try to convert the PDF using `convertPdfText2PdfImage.sh` and try again.

**All scores and feedback will be on Correction folder.**

## Send Feedback

Create a `mail.json` file:

```
./SendMail.py -e > mail.json
```

Edit this file using this template:

```json
{
	"input": "Correction/_scores.csv",
	"delimiter":";",
	"quotechar": "\"",
	"multiple_recipients_separator": ",",

	"sender": "login@server.com",
	"SMTP_server": "smtp.server.com",
	"SMTP_port": "587",
	"SMTP_login": "login@server.com",
//	"SMTP_password": "Your plain password :( ... Leave this commented to ask while running, without storage it."
	"subject": "Your final score is Final_Score",
	"message": """Hi Fullname,
	your score:
		Question 1: Question_1
		Question 2: Question_2
		Question 3: Question_3
		Question 4: Question_4
		Final Score: Final_Score

	Attached, your feedback.

	If you have any questions, please contact me.
""",

	"columns": {
		"email": "EMail",
		"attachment": "Fullname"
	},
	"filter": """def filter(data):
	for header,cell in data.row.items():
		if cell == '':
			return False
	return True
"""
}
```

Change the template above with you SMTP info.

**IMPORTANT:** Use a STARTTLS port.

**TIP:** Create another config file (e.g.: `mail_no_score.json`) and use it to send mail to specific students with no scores (or low scores) just changing the filter code.

# Errors and solutions

**Error:** LaTeX error, no idea.

**Solution:**
1. Create a folder called tex;
1. Include -t tex as parameter;
1. Run `pdftex` manually and see the tex source.

**Error:** There is a %SOMETHING% in LaTeX.

**Solution:** Check the Students.csv header. Some variable is wrong and wasn't replaced.
