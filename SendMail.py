#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
	try:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("config_file", default="mail.json", type=str, nargs='?', help="JSON configuration file (default: mail.json).")
		parser.add_argument("-e", "--example", action="store_true", help="Print a JSON config example.")
		parser.add_argument("-s", "--simulate", action="store_true", help="Just simulate. No email will be sent.")
		args = parser.parse_args()

		if args.example:
			print(r'''{
	"input": "list_students.csv",
	"delimiter":";",
	"quotechar": "\"",
	"multiple_recipients_separator": ",",

	"sender": "login@server.com",
	"SMTP_server": "smtp.server.com",
	"SMTP_port": "587",
	"SMTP_login": "login@server.com",
//	"SMTP_password": "Your plain password :( ... Leave this commented to ask while running, without storage it."
	"subject": "Your final score is HEADER_FINAL_SCORE",
	"message": """Hi HEADER_FULLNAME,

	your score:
		Question 1: HEADER_Q1
		Question 2: HEADER_Q2
		Question 3: HEADER_Q3
		Final Score: HEADER_FINAL_SCORE

	Attached, your feedback.

	If you have any questions, please contact me.
""",

	"columns": {
		"email": "HEADER_EMAIL",
		"attachment": "HEADER_ATTACHMENT"
	},
	"filter": """def filter(data):
	if 'HEADER_SKIP' in data.row.keys() and data.row['HEADER_SKIP'] != '':
		return False # Skip...
	for header,cell in data.row.items():
		if cell == '':
			return False # Skip...
	return True # Send mail
# Others info: 'data.subject' 'data.message' 'data.attachs' 'data.recipients'
"""
}''')
			return


		import chardet
		with open(args.config_file, 'rb') as file:
			row = file.read(32)
			encode = chardet.detect(row)['encoding']
			file.close()

		with open(args.config_file, encoding=encode) as f:
			import json, re, os, collections
			config_path = os.path.dirname(os.path.realpath(args.config_file))

			# Parse Triple Quotes in JSON and convert to array of strings...
			json_str = ""
			triple_quotes = False
			for line in f.read().split("\n"):
				q = line.find("\"\"\"")
				if not triple_quotes:
					row_str =  'r' if q > 0 and line[q-1] == 'r' else ''
					if row_str != '':
						line = line[:q-1]+line[q:]
						q -= 1
				if q >= 0:
					if triple_quotes: # End of triple quotes
						tq_str += ",\n"+row_str+"\"" + line[:q] + "\""
						json_str += tq_str.replace("	","\\t") + "]" + line[q+3:] + "\n"
						triple_quotes = False
					else: # Begin of triple quotes
						q2 = line[q+3:].find("\"\"\"")
						if q2 >= 0: # Begin and End triple quotes in same line
							q2 += q+3
							json_str += line[:q] + "["+row_str+"\"" + line[q+3:q2] + "\"]" + line[q2+3:] + "\n"
						else:
							json_str += line[:q] + "["
							tq_str = ""+row_str+"\"" + line[q+3:] + "\""
							triple_quotes = True
				else:
					if triple_quotes: tq_str   += ",\n"+row_str+"\"" + line + "\""
					else:             json_str += line + "\n"

			# Remove comments
			json_str = re.sub(r"^//.*$", "", json_str, flags=re.M)

			# Allow raw string
			js = ""; mode = 0 # -2: normal quote slash found; -1: inside normal quote; 0: normal; 1: r found; 2: raw quote; 3: raw quote slach found
			for c in range(len(json_str)):
				ch = json_str[c]
				if mode == 0:
					if ch == 'r':
						mode = 1
					elif ch == '"':
						js += ch
						mode = -1
					else:
						js += ch
				elif mode == 1:
					if ch == '"':
						mode = 2
						js += ch
					else:
						js += "r" + ch
						mode = 0
				elif mode == 2:
					if ch == '"':
						js += ch
						mode = 0
					elif ch == "\\":
						mode = 3
					else:
						js += ch
				elif mode == 3:
					if ch == '"':
						js += "\\" + ch
						mode = 2
					elif ch == "\\":
						js += "\\" + ch
					else:
						js += "\\\\" + ch
						mode = 2
				elif mode == -1:
					if ch == "\\":
						mode = -2
					elif ch == '"':
						js += ch
						mode = 0
					else:
						js += ch
				elif mode == -2:
					js += "\\" + ch
					mode = -1

			config = json.loads(js, object_pairs_hook=collections.OrderedDict)
			f.close()

		with open(config['input'], 'rb') as file:
			raw = file.read(32)
			encode = chardet.detect(raw)['encoding']
			file.close()

		rows = []
		with open(config['input'], encoding=encode) as f:
			import csv, collections
			reader = csv.DictReader(f, delimiter=config['delimiter'], quotechar=config['quotechar'])
			for line in reader:
				s = collections.OrderedDict(sorted(line.items(), key=lambda item: reader.fieldnames.index(item[0])))
				rows.append(s)
			f.close()

		dir_input = os.path.dirname(os.path.realpath(config['input']))

		code_str = ""
		for f in config['filter']: code_str += f + "\n"
		def filter(data):
			exec(code_str)
			return eval('filter(data)')
		from collections import namedtuple
		FilterStruct = namedtuple("FilterStruct", "row message subject attachs recipients")

		import smtplib
		import ssl
		from email.mime.application import MIMEApplication
		from email.mime.multipart import MIMEMultipart
		from email.utils import COMMASPACE, formatdate
		from email.mime.text import MIMEText

		if not args.simulate:
			smtp = smtplib.SMTP(config['SMTP_server'], int(config['SMTP_port']))
			context = ssl._create_unverified_context() # DANGER: The certificate will not be verified.
			context.set_ciphers('DEFAULT@SECLEVEL=1') # This will allow for man-in-the-middle attacks and other nasty things.
			smtp.starttls(context=context)
			if "SMTP_password" in config and config["SMTP_password"] != "":
				password = config["SMTP_password"]
			else:
				import getpass
				password = getpass.getpass("Please, enter your SMTP password:")
			smtp.login(config['SMTP_login'], password)

		yes = False
		for r in rows:
			recipients = r[config['columns']['email']].split(config['multiple_recipients_separator'])
			recipients = [r.strip() for r in recipients] # Trim

			msg = MIMEMultipart()
			msg['From'] = config['sender']
			msg['To']   = ", ".join(recipients)
			msg['Date'] = formatdate(localtime=True)

			message = "\n".join(config['message'])
			subject = config['subject']

			attach = None
			if config['columns']['attachment'] in r:
				attach = os.path.join(dir_input, r[config['columns']['attachment']])

			for k,v in r.items():
				message = message.replace(k, v)
				subject = subject.replace(k, v)

			msg['Subject'] = subject
			msg.attach(MIMEText(message))

			attachs = []
			if attach is not None:
				def attach_file(f):
					try:
						with open(f, "rb") as fil:
							part = MIMEApplication(fil.read(),Name=os.path.basename(f))
						part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
						msg.attach(part)
						attachs.append(os.path.basename(f))
					except FileNotFoundError as e:
						print("Warning: file '{}' not found!".format(e.filename))
				path_attach = os.path.realpath(os.path.join(config_path, attach))
				if os.path.isdir(path_attach):
					def attach_rec(d):
						for i in os.listdir(d):
							f=os.path.join(d,i)
							if os.path.isdir(f): attach_rec(f)
							else:                attach_file(f)
					attach_rec(path_attach)
				else:
					attach_file(path_attach)

			data = FilterStruct(row=r, message=message, subject=subject, attachs=attachs, recipients=recipients)
			if filter(data) == False:
				print("Skipping message to {}".format(recipients))
				continue

			print("======")
			print("\tTo:        {}".format(recipients))
			print("\tSubject:   \"{}\"".format(subject))
			print("\tAttachments: {}".format("{} files{}".format(len(attachs), " (" if len(attachs) > 0 else  "." )), end= "")
			for a in range(len(attachs)): print("{}{}".format(attachs[a], ", " if a<len(attachs)-1 else ")."), end="")
			print("")
			print("\tMessage:\n\t\t{}".format("\n\t\t".join(message.split("\n"))))
			print("======")

			send_mail = yes
			if not yes:
				reply = str(input("Send the email above? [y/n/a/q]")).lower().strip()
				if reply == 'y':
					send_mail = True
				elif reply == 'a':
					yes = True
					send_mail = True
				elif reply == 'q':
					exit(0)

			if send_mail:
				print("Sending...", end="")
				if not args.simulate:
					smtp.sendmail(msg['From'], recipients, msg.as_string())
				print(" Successful.")
		if not args.simulate:
			smtp.close()


	except UnicodeDecodeError as e:
		import traceback
		traceback.print_exc()
		print('Try to save the JSON configuration file using UTF-8 with BOM.')

	except Exception as e:
		import traceback
		traceback.print_exc()

if __name__ == "__main__":
	main()
