# Manual

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
