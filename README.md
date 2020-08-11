# MakeTests

## First Test

### Create a Question's DataBase

```
# Make directoreis and subdirectories
mkdir Questions
mkdir Questions/Easy
mkdir Questions/Medium
mkdir Questions/Hard

# Create some questions from template
./MakeTests.py -e choices > Questions/Easy/choices.py
./MakeTests.py -e truefalse > Questions/Easy/truefalse.py
./MakeTests.py -e questionanswer > Questions/Medium/questionanswer.py
./MakeTests.py -e number > Questions/Medium/number.py
./MakeTests.py -e essay > Questions/Hard/essay.py
```

Edit/add/delete your questions.

### Create a config file

```
./MakeTests.py -e config > config.json
```

In question section, define path to data base.
```
"db_path": "Questions"
```
Select questions
```
"select": [
  {"path": "Easy",  "weight": 1, "replaces": {"%PREFIX%": "Weight 1"}},
  {"path": "Medium",  "weight": 2, "replaces": {"%PREFIX%": "Weight 2"}},
  {"path": "Medium",  "weight": 2, "replaces": {"%PREFIX%": "Weight 2"}},
  {"path": "Hard",  "weight": 1, "replaces": {"%PREFIX%": "Weight 2"}}
]
```




Create a CSV student's list file called students.csv. Example:

```
ID,Name,EMail
000001; "Alice"; alice@maketests.com
000002; "Bob"; bob@maketests.com
```
