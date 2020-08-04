#!/usr/bin/env python3
# -*- coding: utf-8 -*-


## TIPS
#	MACPORT
#		Install MacPorts https://www.macports.org
#		Execute in Terminal:
#			sudo port -v selfupdate
#		
#	ZBAR
#		MAC OSX:
#			xcode-select --install
#			ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#			brew install zbar
#		LINUX:
#			sudo apt-get install libzbar-dev libzbar0
#
#	CYTHON
#		pip install Cython --install-option="--no-cython-compile"
#
#	PYMINIFIER
#		sudo python -m pip install pyminifier
#

"""
class Question:
	info							dict()   | Information about the current student.
	def makeVariables(self):        -> None  | Generate randomically all variables of the question.
	def getQuestionTex(self):       -> str   | Return the questions description, in LaTeX.
	def answerAreaAspectRate(self): -> float | Aspect rate of Answer Area desired (width/height).
	def drawAnswerArea(self, img):  -> img   | Draw using cv2 an unfilled area answer in img.
	def doCorrection(self, img):    -> str   | Correct img answer and return the score.
	def getAnswerText(self,LaTeX):  -> str   | Return the answer in LaTeX or Plain Text, used to template.

class QuestionMatrix(Question):
	rows                              -> [str]                    | List of row's label  (Set it at makeVariable())
	cols                              -> [str]                    | List of cols's label  (Set it at makeVariable())
	hlabel                            -> str                      | Horizontal label. None to disable.  (Set it at makeVariable())
	vlabel                            -> str                      | Vertical label. None to disable.  (Set it at makeVariable())
	def getScore(self, matrix_answer) -> str                      | Return the score using matrix_answer.
	def getTemplate(self)             -> matrix[boolean][boolean] | Return the correct boolean matrix.
	IMPLEMENTED: answerAreaAspectRate; drawAnswerArea; doCorrection
	TO IMPLEMENT: makeVariables; getQuestionText; getAnswerText
"""



import sys
req_version = (3,0)
cur_version = sys.version_info
if cur_version < req_version:
	sys.exit("Your Python interpreter is old (version {}.{}). The minimum versioin required is Python {}.{}.".format(cur_version[0],cur_version[1],req_version[0],req_version[1]))

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
	pass
elif _platform == "darwin":
	pass
elif _platform == "win32":
	sys.exit("Windows has not yet been tested.")

try:
	import conda, pip, cv2, qrcode, pyzbar, barcode, PyPDF2, pytesseract, screeninfo
except ImportError as error:
	if error.name == "conda":
		sys.exit("Please, install conda (https://www.anaconda.com/download).")
	else:
		package_name = {"pip":         ["conda", "install", "-c", "anaconda",    "pip"],
#                        "cv2":         ["conda", "install", "-c", "menpo",       "opencv3"],
                        "cv2":         ["pip",   "install",                      "opencv-python"],
                        "qrcode":      ["conda", "install", "-c", "conda-forge", "qrcode"],
                        "PyPDF2":      ["conda", "install", "-c", "conda-forge", "pypdf2"],
                        "tesseract":   ["conda", "install", "-c", "conda-forge", "tesseract"],
                        "pyzbar":      ["pip",   "install",                      "pyzbar"],
                        "barcode":     ["pip",   "install",                      "python-barcode"],
                        "pytesseract": ["pip",   "install",                      "pytesseract"],
                        "screeninfo":  ["pip",   "install",                      "screeninfo"]}
		print(error.msg)
		if error.name not in package_name:
			sys.exit("Please, install {} library...".format(error.name))
		else:
			reply = str(input("Do you want try to install {} now with command '{}'? [y/n]".format(error.name, " ".join(package_name[error.name])))).lower().strip()
		if reply[:1] == 'y':
			import subprocess
			subprocess.call(["sudo", sys.executable, "-m"] + package_name.get(error.name, [error.name]))
		sys.exit()



###############
# BEGIN UTILS #
class Utils:
	@staticmethod
	def loadModules(relative_path):
		import os
		return Utils.loadModulesAbs(os.path.realpath(os.path.join(os.getcwd(), relative_path)))

	@staticmethod
	def loadModulesAbs(path):
		import os, sys
		if not os.path.isdir(path):
			raise Exception("Directory '{}' didn't exist.".format(path))
		modules = dict()
		for x in os.listdir(path):
			if os.path.isfile(os.path.join(path, x)):
				if x.endswith(".py"): # New module found!
					from importlib import import_module
					module_name = os.path.splitext(x)[0] # Remove extension '.py'.
					sys.path.insert(0, path)
					modules[module_name] = import_module(module_name)
					sys.path.pop(0)
					del sys.modules[module_name] # Removing module, in case of future repeated module name.
				else: # Other file type.
					pass # Ignore it...
			else: # Is directory.
				tmp = Utils.loadModulesAbs(os.path.join(path, x))
				if len(tmp) > 0: # Ignore if there is no modules.
					modules[x] = tmp
		import collections # standard dict() is unordered. Using OrderedDict.
		return collections.OrderedDict(sorted(modules.items(), key=lambda t: t[0]))

	@staticmethod
	def json2dict(s):
		import json, re
		from collections import OrderedDict
		try:
			inc = ""

			# Parse Tiple Quotes in JSON and convert to array of strings...
			json_str = ""
			triple_quotes = False
			for line in s.split("\n"):
				q = line.find("\"\"\"")
				if not triple_quotes:
					raw_str =  'r' if q > 0 and line[q-1] == 'r' else ''
					if raw_str != '':
						line = line[:q-1]+line[q:]
						q -= 1
				if q >= 0:
					if triple_quotes: # End of triple quotes
						tq_str += ",\n"+raw_str+"\"" + line[:q] + "\""
						json_str += tq_str.replace("	","\\t") + "]" + line[q+3:] + "\n"
						triple_quotes = False
					else: # Begin of triple quotes
						q2 = line[q+3:].find("\"\"\"")
						if q2 >= 0: # Begin and End triple quotes in same line
							q2 += q+3
							json_str += line[:q] + "["+raw_str+"\"" + line[q+3:q2] + "\"]" + line[q2+3:] + "\n"
						else:
							json_str += line[:q] + "["
							tq_str = ""+raw_str+"\"" + line[q+3:] + "\""
							triple_quotes = True
				else:
					if triple_quotes: tq_str   += ",\n"+raw_str+"\"" + line + "\""
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

			data = json.loads(js, object_pairs_hook=OrderedDict)
			if 'includeJSON' in data:
				from copy import deepcopy
				includes = deepcopy(data['includeJSON'])
				for inc in includes:
					with open(inc) as f:
						inc_dict = Utils.json2dict(f.read())
						Utils.jsonMerge(inc_dict, data, raiseErrorFromUnexpected=False)
						f.close()
					data = inc_dict
			return data
		except FileNotFoundError as e:
			raise Exception("Config file '{}' not found!".format(e.filename))
		except Exception as e:
			if inc == "":
				raise Exception("Parser error: {}".format(e))
			else:
				raise Exception("Parser error in {}: {}".format(inc, e))

	@staticmethod
	def jsonMerge(json_base, json_extra, raiseErrorFromUnexpected):
		for k,v in json_extra.items():
			if   k[0 ] == '+': k = k[1: ]; attach = 1 # attach begin
			elif k[-1] == '+': k = k[:-1]; attach = 2 # attach end
			else:                          attach = 0 # no attach
			if k not in json_base:
				if raiseErrorFromUnexpected:
					raise Exception("Try to insert an unexpected key: '{}'.".format(k))
				else:
					json_base[k] = v
			elif isinstance(v,dict):
				Utils.jsonMerge(json_base[k],json_extra[k], raiseErrorFromUnexpected=raiseErrorFromUnexpected)
			elif isinstance(v,list):
				if attach == 0:
					json_base[k] = v
				elif attach == 1:
					v.extend(json_base[k])
					json_base[k] = v
				elif attach == 2:
					json_base[k].extend(v)
			else:
				json_base[k] = v

	@staticmethod
	def str2intHash(s, digits=16):
		import hashlib
		return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**digits

	@staticmethod
	def str2orderedDict(s):
		from collections import OrderedDict
		from ast import literal_eval
		file_list=literal_eval(s[13:-2])
		header=OrderedDict()
		for entry in file_list:
			key, value = entry
			header[key] = value
		return header

	@staticmethod
	def getImagesFromPDF(pdf_filename):
		import PyPDF2 as pypdf
		import numpy as np
		from PIL import Image
		pdf = pypdf.PdfFileReader(open(pdf_filename, "rb"))
		for pg in range(pdf.getNumPages()):
			page = pdf.getPage(pg)
			objs = page['/Resources']['/XObject'].getObject()
			for obj in objs:
				if objs[obj]['/Subtype'] == '/Image':
					if objs[obj]['/Filter'] == '/DCTDecode':
						yield cv2.imdecode(np.frombuffer(objs[obj]._data, np.uint8), cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE
					elif objs[obj]['/Filter'] == '/FlateDecode':
						size = (objs[obj]['/Width'],objs[obj]['/Height'])
						mode = "RGB" if objs[obj]['/ColorSpace'] == '/DeviceRGB' else "P"
						img = Image.frombytes(mode, size, objs[obj].getData())
						yield ImageUtils.pil2opencv(img)
					elif type(objs[obj]['/Filter']) is PyPDF2.generic.ArrayObject:
						for f in objs[obj]['/Filter']:
							if f == '/DCTDecode':
								yield cv2.imdecode(np.frombuffer(objs[obj]._data, np.uint8), cv2.IMREAD_COLOR)
							else:
								raise Exception("Filter {} didn't implemented yet!".format(objs[obj]['/Filter']))
					else:
						raise Exception("Filter {} didn't implemented yet!".format(objs[obj]['/Filter']))

	@staticmethod
	def getEncodeFile(filename):
		import chardet
		with open(filename, 'rb') as file:
			raw = file.read(32)
		return chardet.detect(raw)['encoding']
# END UTILS #
#############



#########################
# BEGIN IMAGE PROCESSOR #
class ImageUtils:
	SCREEN_RESOLUTION   = (2000,2000) # TODO: Calculate width and height of screen. Used just to show image.
	MAX_IMAGE_RES       = (2048,3072)
	IMAGE_WIDTH         = 1024
	IMAGE_HEADER_HEIGHT = 45
	IMAGE_BORDER        = 2
	IMAGE_UP_CODE       = "0"
	IMAGE_DOWN_CODE     = "1"
	BARCODE_TYPE        = 'code128'

	@staticmethod
	def makeAnswerArea(code, aspectrate):
		import cv2, numpy as np

		# Calculating variables
		width         = ImageUtils.IMAGE_WIDTH
		header_height = ImageUtils.IMAGE_HEADER_HEIGHT
		border        = ImageUtils.IMAGE_BORDER
		footer_height = header_height
		marker_radius = header_height // 2
		padding       = marker_radius
		height        = header_height + padding + int(width/aspectrate) + padding + footer_height

		# Make a blank answer area image
		img_border = np.zeros((height+2*border, width+2*border, 3), np.uint8)
		img_border[:,:] = (255,255,255)
		img = img_border[border:border+height,border:border+width]

		# Put barcode in top bar
		c1 = ImageUtils.pil2opencv(ImageUtils.makeBarcode(ImageUtils.IMAGE_UP_CODE + code, typeBarcode=ImageUtils.BARCODE_TYPE))
		c1 = cv2.resize(c1, (int(width - 4*marker_radius), header_height))
		ImageUtils.overlayImage(img, c1, (0, width//2 - c1.shape[1]//2) )

		# Put barcode in bottom bar
		c2 = ImageUtils.pil2opencv(ImageUtils.makeBarcode(ImageUtils.IMAGE_DOWN_CODE + code, typeBarcode=ImageUtils.BARCODE_TYPE))
		c2 = cv2.resize(c2, (int(width - 4*marker_radius), footer_height))
		ImageUtils.overlayImage(img, c2, (height-footer_height, width//2 - c2.shape[1]//2) )

		# Draw markers
		ImageUtils.drawMarker(img_border, (border+      marker_radius,border+       header_height-marker_radius), marker_radius)
		ImageUtils.drawMarker(img_border, (border+width-marker_radius,border+       header_height-marker_radius), marker_radius)
		ImageUtils.drawMarker(img_border, (border+      marker_radius,border+height-footer_height+marker_radius), marker_radius)
		ImageUtils.drawMarker(img_border, (border+width-marker_radius,border+height-footer_height+marker_radius), marker_radius)

		return img_border, img[header_height+padding:header_height+padding+int(width/aspectrate),:]

	@staticmethod
	def findAnswerAreas(img, verbose=False):
		import numpy as np

		# Find all bars (header or footer of an answer area)
		bars = []
		markers = [m for m in ImageUtils.markerDetector(img)]
		for i1 in range(len(markers)):
			c1, r1 = markers[i1] 
			if verbose:
				cv2.circle(img, (int(c1[0]),int(c1[1])), int(r1),(0,0,255), thickness=cv2.FILLED)
			for i2 in range(i1+1,len(markers)):
				c2,r2 = markers[i2] 
				p1, p2 = ImageUtils.findPointsPerpendicularToTheLine(c1, c2, r1)
				p3, p4 = ImageUtils.findPointsPerpendicularToTheLine(c2, c1, r2)
				box = np.int0([p1, p2, p3, p4])
				warp = ImageUtils.warpImage(img, box)
				for p,d,t in ImageUtils.barcodeDecoder(warp):
					if verbose:
						cv2.drawContours(img,[box],0,(0,0,255),2)
					bars.append([i1, i2, d])

		# Find all sections (headers and footers bars of a same answer area)
		sections = []
		for i1 in range(len(bars)):
			_, _, d1 = bars[i1]; d1 = d1.decode('utf-8')
			for i2 in range(i1+1, len(bars)):
				_, _, d2 = bars[i2]; d2 = d2.decode('utf-8')
				if ( (d1[0] == ImageUtils.IMAGE_UP_CODE   and d2[0] == ImageUtils.IMAGE_DOWN_CODE) or
				     (d1[0] == ImageUtils.IMAGE_DOWN_CODE and d2[0] == ImageUtils.IMAGE_UP_CODE  ) ) and d1[1:] == d2[1:]:
					if d1[0] == ImageUtils.IMAGE_UP_CODE:
						sections.append([bars[i1], bars[i2], str(d1[1:])])
					else:
						sections.append([bars[i2], bars[i1], str(d1[1:])])

		# Find answer area
		for topBar, bottomBar, code in sections:
			tl = markers[topBar[0]][0];    tlr = markers[topBar[0]][1]
			tr = markers[topBar[1]][0];    trr = markers[topBar[1]][1]
			bl = markers[bottomBar[0]][0]; blr = markers[bottomBar[0]][1]
			br = markers[bottomBar[1]][0]; brr = markers[bottomBar[1]][1]

			# Correcting possible reversals coordinates
			# TODO: Use imutils sort_contours ? (https://www.pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/)
			if tl[0] > tr[0]:
				tl, tr, tlr, trr = tr, tl, trr, tlr
			if bl[0] > br[0]:
				bl, br, blr, brr = br, bl, brr, blr
			if tl[1]+tr[1] > bl[1]+br[1]: # Check if upside down
				if tl[0] < tr[0]:
					tl, tr, tlr, trr = tr, tl, trr, tlr
				if bl[0] < br[0]:
					bl, br, blr, brr = br, bl, brr, blr

			# Store markers info to yield it later
			markers_info = {"tl": tl, "tlr": tlr, "tr": tr, "trr": trr, "bl": bl, "blr": blr, "br": br, "brr": brr}

			# Adjust coordinates to fit the Answer Area
			tl = ImageUtils.findPointAlongTheLine(tl, bl, tlr*2.0); tr = ImageUtils.findPointAlongTheLine(tr, br, trr*2.0)
			bl = ImageUtils.findPointAlongTheLine(bl, tl, blr*2.0); br = ImageUtils.findPointAlongTheLine(br, tr, brr*2.0)
			tl = ImageUtils.findPointAlongTheLine(tl, tr, -tlr   ); tr = ImageUtils.findPointAlongTheLine(tr, tl, -trr   )
			bl = ImageUtils.findPointAlongTheLine(bl, br, -blr   ); br = ImageUtils.findPointAlongTheLine(br, bl, -brr   )

			# Get answer area
			ansArea = ImageUtils.warpImage(img, (tl,tr,br,bl))
			if verbose:
				cv2.drawContours(img, [np.int0([tl,tr,br,bl])],0,(0,255,255), thickness=2)

			# Normalize the width of the Answer Area
			h, w, _ = ansArea.shape
			ansArea = cv2.resize(ansArea, (ImageUtils.IMAGE_WIDTH,int(ImageUtils.IMAGE_WIDTH*h/w)))

			yield ansArea, code, [np.int0([tl,tr,br,bl])], markers_info

	@staticmethod
	def makeBarcode(data, typeBarcode='code128'):
		import barcode
		ean = barcode.get(name=typeBarcode, code=data, writer=barcode.writer.ImageWriter())
		im = ean.render()
		return im.crop(box=(0, 0, im.width, im.height//2)) # Remove numbers below

	@staticmethod
	def makeQRCode(data):
		import qrcode
		qr = qrcode.QRCode(
			version=1,
			error_correction=qrcode.constants.ERROR_CORRECT_L,
			box_size=4,
			border=0
		)
		qr.add_data(data)
		qr.make(fit=True)
		return qr.make_image(fill_color="black", back_color="white")

	@staticmethod
	def drawTextInsideTheBox(i, t, color=(0,0,0), thickness=2):
		h,w,_ = i.shape
		argText = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "color":color, "thickness": thickness, "lineType": cv2.LINE_AA, "bottomLeftOrigin": False}
		argTextSize = {"fontFace": argText['fontFace'], "thickness": argText['thickness']}
		(tex_w, tex_h), bl = cv2.getTextSize(t, fontScale=1, **argTextSize)
		fs_w = w/tex_w; fs_h = h/(bl+tex_h)
		argText['fontScale'] = fs_w if fs_w < fs_h else fs_h; argTextSize['fontScale'] = argText['fontScale']
		(tex_w, tex_h), bl = cv2.getTextSize(t, **argTextSize)
		cv2.putText(i, t, (w//2-tex_w//2,h//2+tex_h//2-bl//2), **argText)

	@staticmethod
	def drawMarker(img, pos, radius):
		args = {"img": img, "thickness":cv2.FILLED, "shift":0, "lineType": cv2.LINE_AA}
		cv2.circle(center=pos, radius=int(radius*1.00), color=(  0,  0,  0), **args)
		cv2.circle(center=pos, radius=int(radius*0.75), color=(255,255,255), **args)
		cv2.circle(center=pos, radius=int(radius*0.47), color=(  0,  0,  0), **args)
		cv2.circle(center=pos, radius=int(radius*0.20), color=(255,255,255), **args)

	@staticmethod
	def barcodeDecoder(img):
		import numpy as np
		import pyzbar.pyzbar as pyzbar
		dec = pyzbar.decode(cv2.GaussianBlur(img, (3,3), 0))
		for c in dec:
			box = cv2.approxPolyDP(np.array(c.polygon), 0.1*cv2.arcLength(np.array(c.polygon), True), True)
			if len(box) > 0:
				yield (np.array([b[0] for b in box]), c.data, c.type)

	@staticmethod
	def qrCodeDecoder(img):
		import numpy as np
		import pyzbar.pyzbar as pyzbar
		dec = pyzbar.decode(cv2.GaussianBlur(img, (3,3), 0))
		for c in dec:
			if c.type == 'QRCODE':
				yield (np.array(c.polygon), c.data)

	@staticmethod
	def markerDetector(img, minHierarchy=7, smallArea=False):
		border = 2
		# http://dsynflo.blogspot.com/2014/10/opencv-qr-code-detection-and-extraction.html
		import numpy as np
		img = cv2.copyMakeBorder(img, top=border, bottom=border, left=border, right=border, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
		for countour in ImageUtils.markerContours(img, minHierarchy=minHierarchy, smallArea=smallArea):
			points = []
			for p in countour:
				points.append([p[0][0]-border, p[0][1]-border])
			center, radius = cv2.minEnclosingCircle(np.array(points))
			yield center, radius * 0.9

	@staticmethod
	def markerContours(img, minHierarchy, smallArea):
		# https://gist.github.com/jas0n1ee/09ddc418d0e5b6189f3cf8f63678c858
		import cv2
		import numpy as np
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if smallArea: # For smalls regions, use threshold image in canny is better...
			gray = cv2.GaussianBlur(gray, (5,5), 0)
			gray = cv2.medianBlur(gray, 3)
			max_thresh_val, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
			thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
			edges = cv2.Canny(thresh_img,threshold1=max_thresh_val//2,threshold2=max_thresh_val,apertureSize=3)
		else:
			r = max(img.shape[0]/ImageUtils.MAX_IMAGE_RES[1], img.shape[1]/ImageUtils.MAX_IMAGE_RES[0])
			if   r >= 0.99: ks1 = 7; ks2 = 9; ks3 = 6
			elif r >= 0.75: ks1 = 5; ks2 = 7; ks3 = 4
			elif r >= 0.50: ks1 = 3; ks2 = 5; ks3 = 2
			else:           ks1 = 3; ks2 = 3; ks3 = 1

			el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks3,ks3))
			gray = cv2.GaussianBlur(gray, (ks2,ks2), 0)
			gray = cv2.medianBlur(gray, ks1)
			gray = cv2.bilateralFilter(gray,ks2,75,75)

			# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
			sigma = 0.33; v = np.median(img); lower = int(max(0, (1.0 - sigma) * v)); upper = int(min(255, (1.0 + sigma) * v))
			thresh_img = cv2.Canny(gray, lower, upper,apertureSize=3)
			edges = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, el)

		_, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		tot = 0
		if len(contours):
			hierarchy = hierarchy[0]
		for i in range(len(contours)):
			k = i
			c = 0
			while hierarchy[k][2] != -1:
				k = hierarchy[k][2]
				c = c + 1
			if hierarchy[k][2] != -1:
				c = c + 1
			if c >= minHierarchy:
				tot+=1
				yield contours[i]
		if __debug__:
			gray = np.hstack((img, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))
			gray2 = np.hstack((cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB),cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)))
			gray = np.vstack((gray, gray2))
			if tot == 0: cv2.imshow(__class__.__name__ + "." + sys._getframe(1).f_code.co_name + "_notFound", ImageUtils.conditionalResize(gray))
			else:        cv2.imshow(__class__.__name__ + "." + sys._getframe(1).f_code.co_name              , ImageUtils.conditionalResize(gray))

	@staticmethod
	def pil2opencv(img):
		import numpy as np
		i = np.array(img.convert('RGB'))
		return cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
	
	@staticmethod
	def opencv2pil(img):
		from PIL import Image
		import numpy as np
		return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	
	@staticmethod
	def rotateBound(image, angle):
		# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
		import numpy as np
		(h, w) = image.shape[:2]
		(cX, cY) = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
		cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
		nW = int((h * sin) + (w * cos)); nH = int((h * cos) + (w * sin))
		M[0, 2] += (nW / 2) - cX; M[1, 2] += (nH / 2) - cY
		return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_REFLECT)

	@staticmethod
	def overlayImage(big, small, pos):
		h, w, _ = small.shape
		big[ pos[0] : pos[0]+h , pos[1] : pos[1]+w ] = small

	@staticmethod
	def overlayWarpImage(big, small, quad_big, quad_small):
		import numpy as np
		M = cv2.getPerspectiveTransform(quad_small, quad_big)
		i = cv2.warpPerspective(small, M, (big.shape[1], big.shape[0]))
		cv2.drawContours(big, [np.array([(x[0],x[1]) for x in quad_big[0]], np.int32)], 0, (0,0,0), -1)
		big[:] = big[:] + i[:]

	@staticmethod
	def warpImage(img, rect):
		# https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
		import numpy as np
		(tl, tr, br, bl) = rect
		maxWidth = max(int(ImageUtils.distance(br, bl)), int(ImageUtils.distance(tr, tl)))
		maxHeight = max(int(ImageUtils.distance(tr, br)), int(ImageUtils.distance(tl, bl)))
		dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
		M = cv2.getPerspectiveTransform(np.float32(rect), dst)
		return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

	@staticmethod
	def findPointsPerpendicularToTheLine(lineP1, lineP2, distance):
		# https://stackoverflow.com/questions/133897/how-do-you-find-a-point-at-a-given-perpendicular-distance-from-a-line
		from math import sqrt
		dx = lineP1[0]-lineP2[0]; dy = lineP1[1]-lineP2[1]; dist = sqrt(dx*dx + dy*dy)
		dx /= dist; dy /= dist
		return [[lineP1[0] + (distance)*dy, lineP1[1] - (distance)*dx], [lineP1[0] - (distance)*dy, lineP1[1] + (distance)*dx]]

	@staticmethod
	def findPointAlongTheLine(lineP1, lineP2, distance):
		from math import sqrt
		dx = lineP1[0]-lineP2[0]; dy = lineP1[1]-lineP2[1]; dist = sqrt(dx*dx + dy*dy)
		ratio = distance/dist
		return [(1-ratio)*lineP1[0] + ratio*lineP2[0], (1-ratio)*lineP1[1] + ratio*lineP2[1]]

	@staticmethod
	def conditionalResize(img, max_res = None):
		if max_res is None: max_res = ImageUtils.SCREEN_RESOLUTION
		cur_res = (img.shape[1], img.shape[0])
		if max_res[0] < cur_res[0] or max_res[1] < cur_res[1]:
			if max_res[0]/max_res[1] < cur_res[0]/cur_res[1]: new_res = (max_res[0], int(max_res[0]*cur_res[1]/cur_res[0]))
			else:                                             new_res = (int(max_res[1]*cur_res[0]/cur_res[1]), max_res[1])
			img = cv2.resize(img, new_res, img, interpolation=cv2.INTER_LANCZOS4)
		return img

	@staticmethod
	def distance(p1, p2):
		from scipy.spatial import distance as dist
		return dist.euclidean(p1, p2)

	@staticmethod
	def midPoint(p1, p2):
		return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
# BEGIN IMAGE PROCESSOR #
#########################



#############
# BEGIN LATEX #
class LaTeX:
	replaces = None
	includes = None
	tex      = None
	temp_dir = None
	proc_out = None
	img_counter = 0

	def __init__(self, sufix_temp_dir="", temporary_directory=None):
		import os, tempfile, collections
		
		if temporary_directory is None:
			self.tempfile_TemporaryDirectory = tempfile.TemporaryDirectory(prefix=os.path.splitext(os.path.basename(__file__))[0], suffix=sufix_temp_dir)
			self.temp_dir = self.tempfile_TemporaryDirectory.name
		else:
			self.temp_dir = tempfile.mkdtemp(prefix=os.path.splitext(os.path.basename(__file__))[0], suffix=sufix_temp_dir, dir=os.path.join(os.path.dirname(os.path.realpath('__file__')), temporary_directory))

		self.replaces = collections.OrderedDict()
		self.includes = []
		self.tex      = []

	def addReplaces(self, replaces):
		for k, v in replaces.items():
			if type(k) is not str or type(v) is not str:
				raise Exception("LaTeX.addReplaces dict must be a str:str!")
			self.replaces[k] = v.replace("_","\_")

	def addInclude(self, inc):
		import os
		if type(inc) is not str:
			raise Exception("LaTeX.addInclude argument must be a str!")
		elif not os.path.isdir(inc):
			raise Exception("LaTeX.addInclude is trying to include an invalid directory (\"{}\")".format(inc))
		self.includes.append(inc)

	def addImage(self, img, prefix="", sufix="", extension=".png", noIncrement=False, save=True):
		import os
		# Make an unique path to image file
		while(True):
			file_name = prefix + str(self.img_counter) + sufix + extension
			full_path = os.path.join(self.temp_dir,file_name)
			# If already exist, try other...
			if os.path.exists(full_path): self.img_counter += 1
			else:                         break
		if save:
			try: # Pil Image?
				f = open( full_path, 'wb' )
				img.save(f); f.flush(); f.close() # Save now! No cache...
			except AttributeError:
				import cv2
				try: # OpenCV Image?
					cv2.imwrite(full_path, img)
				except TypeError as e:
					try: # matplotlib.pyplot?
						img.savefig(full_path)
					except AttributeError as e:
						raise AttributeError(e)
		return full_path

	def addTex(self, tex):
		if type(tex) is str:
			for k, v in self.replaces.items():
				tex = tex.replace(k, v)		
			lines = tex.split('\n')
			for l in lines:
				self.tex.append(l)
		elif type(tex) is list:
			for i in tex:
				self.addTex(i)
		else:
			raise Exception("LaTeX.addTex argument must be a str or a list of str!")

	def makePdf(self, output, verbose=True):
		self.proc_out = ""
		for l in LaTeX.latex2pdf(self.tex, output, includes=self.includes, tmp_dir=self.temp_dir):
			self.proc_out += l
			yield l

	def printTex(self):
		for i in range(len(self.tex)):
			print("{:5d}: {}".format(i+1, self.tex[i]))

	def getError(self):
		err_lines = self.proc_out.split('\n')
		for l in range(len(err_lines)):
			line = err_lines[l]
			if (line.find("Error") >= 0 or line.find("error") >= 0) and line.find("./source.tex:") >= 0:
				import re
				err_line = re.findall(r'\d+', line)
				if len(err_line) > 0:
					err_line = int(err_line[0])
					ret, offset = "", 2
					for i in range(max(err_line-offset, 1),min(err_line+offset, len(self.tex))+1):
						ret += "{}{:5d}: {}{}".format(" " if i != err_line else ">", i, self.tex[i-1], "\n" if i < min(err_line+offset, len(self.tex)) else "")
					for i in range(l+1, len(err_lines)):
						if err_lines[i] != "": line += err_lines[i]
						else:                   break
					return line + "\n" + ret
		return self.proc_out

	@staticmethod
	def latex2pdf(tex_str, output, tmp_dir, includes=[]):
		import os, sys, subprocess, shlex, shutil, errno

		# Get full output path
		full_path_output = os.path.join(os.path.dirname(os.path.realpath('__file__')), output)

		# Check if temp directory exist
		if not os.path.exists(tmp_dir):
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), tmp_dir)

		# Link directories from incluldes
		for i in includes:
			os.symlink(os.path.join(os.path.dirname(os.path.realpath('__file__')),i), os.path.join(tmp_dir, os.path.basename(i))) # shutil.copytree

		# Set working directory to temp directory
		cwd = os.getcwd()
		os.chdir(tmp_dir)

		# Create source.tex
		filename = 'source'
		with open(filename + '.tex', 'w') as f:
			for line in tex_str:
				f.write(str(line))
				f.write('\n')
			f.close()

		# Compile LaTeX
		cmd = [ "pdflatex", "-halt-on-error", "-file-line-error", "-output-format=pdf", filename + '.tex' ]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

		# Get output
		try:
			for stdout_line in iter(proc.stdout.readline, ""):
				yield stdout_line
		except UnicodeDecodeError as e:
			proc.kill()
			raise e

		# Check return
		return_code = proc.wait()
		if return_code:
			log = open(filename + '.log').read()
			raise subprocess.CalledProcessError(proc.returncode, proc.args, log)

		# Move output file
		if os.path.isfile(filename + '.pdf'):
			shutil.move(filename + '.pdf', full_path_output)

		# Restore current working directory
		os.chdir(cwd)
# END LATEX #
###########



##################
# BEGIN QUESTION #
#  ABSTRACT !!!  #
class Question:
	info = None
	addImage = None
	def __init__(self, **info):
		self.info = info
	def makeVariables(self):
		raise Exception("Method not implemented.")
	def getQuestionTex(self):
		raise Exception("Method not implemented.")
	def answerAreaAspectRate(self):
		raise Exception("Method not implemented.")
	def drawAnswerArea(self, img):
		raise Exception("Method not implemented.")
	def doCorrection(self, img):
		raise Exception("Method not implemented.")
	def getAnswerText(self, LaTeX=True):
		raise Exception("Method not implemented.")
# END QUESTION #
################



#########################
# BEGIN QUESTION MATRIX #
class QuestionMatrix(Question):
	RADIUS_CIRCLE_FACTOR = 3   # radius_circle_answer = min(width_cell,height_cell) / RADIUS_CIRCLE_FACTOR
	MAX_RADIUS_SIZE      = 80  # radius_circle_answer = image_width / MAX_RADIUS_SIZE
	CIRCLE_THICKNESS     = 2

	rows   = [chr(ord("A") + i) for i in range(  5  )]
	cols   = [str(i) for i in range(1,1+  5  )]
	hlabel = None
	vlabel = None

	def getScore(self, matrix_answer):
		raise Exception("Method not implemented.")
	def getTemplate(self):
		raise Exception("Method not implemented.")

	def answerAreaAspectRate(self):
		return (len(self.cols)+(1 if self.vlabel is None else 2)) / (len(self.rows)+(1 if self.hlabel is None else 2))

	def drawAnswerArea(self, img):
		import numpy as np
		height, width, _ = img.shape
		v_cells = len(self.cols)+(1 if self.vlabel is None else 2)
		h_cells = len(self.rows)+(1 if self.hlabel is None else 2)
		cell_w = width / v_cells; cell_h = height/ h_cells
		rad = int(cell_w//QuestionMatrix.RADIUS_CIRCLE_FACTOR if cell_w<cell_h else cell_h//QuestionMatrix.RADIUS_CIRCLE_FACTOR)
		if rad > width/QuestionMatrix.MAX_RADIUS_SIZE: rad = int(width/QuestionMatrix.MAX_RADIUS_SIZE)

		if self.hlabel is not None:
			ImageUtils.drawTextInsideTheBox(img[0:int(cell_h),0:int(width-cell_w)], self.hlabel)

		if self.vlabel is not None:
			vLabelImg = np.zeros((int(cell_w), int(height-cell_h), 3), np.uint8)
			vLabelImg[:,:] = (255,255,255)
			ImageUtils.drawTextInsideTheBox(vLabelImg, self.vlabel)
			ImageUtils.overlayImage(img, ImageUtils.rotateBound(vLabelImg, 90), (int(cell_h),int(width-cell_w)))

		for r in range(len(self.rows)):
			(i,j) = ((1 if self.hlabel is None else 2)+r)*cell_h, (len(self.cols))*cell_w
			ImageUtils.drawTextInsideTheBox(img[int(i):int(i+cell_h), int(j):int(j+cell_w)], self.rows[r])

		for c in range(len(self.cols)):
			(i,j) = (0 if self.hlabel is None else 1)*cell_h, c*cell_w
			ImageUtils.drawTextInsideTheBox(img[int(i):int(i+cell_h), int(j):int(j+cell_w)], self.cols[c])

		for r in range(len(self.rows)):
			i = ((1 if self.hlabel is None else 2)+r)*cell_h
			for c in range(len(self.cols)):
				j = c*cell_w
				cv2.circle(img, center=(int(j+cell_w/2),int(i+cell_h/2)), radius=rad, color=(0,0,0), thickness=self.CIRCLE_THICKNESS, lineType=cv2.LINE_AA, shift=0)

	def doCorrection(self, img):
		import numpy as np
		height, width, _ = img.shape
		v_cells = len(self.cols)+(1 if self.vlabel is None else 2)
		h_cells = len(self.rows)+(1 if self.hlabel is None else 2)
		cell_w = width / v_cells; cell_h = height/ h_cells
		rad = int(cell_w//QuestionMatrix.RADIUS_CIRCLE_FACTOR if cell_w<cell_h else cell_h//QuestionMatrix.RADIUS_CIRCLE_FACTOR)
		if rad > width/QuestionMatrix.MAX_RADIUS_SIZE: rad = int(width/QuestionMatrix.MAX_RADIUS_SIZE)

		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.medianBlur(img, 3); img = cv2.GaussianBlur(img, (5,5),0)
		img_gray = cv2.medianBlur(img_gray, 3); img_gray = cv2.GaussianBlur(img_gray, (5,5),0)
		img_gray = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,2*((rad*4)//2)+1,C=2)

		# Get the correct answers
		template = self.getTemplate()

		# Make matrix (ans) with True or False (cheked and unchecked)
		offset  = 0.5 # neighborhood inclusion
		border  = 0.0 # the centers needs to be inside box plus border
		rad_min = 0.7 * rad
		rad_max = 1.8 * rad + 2 * self.CIRCLE_THICKNESS
		all_successful = True
		ans = []
		for u in range((1 if self.hlabel is None else 2), h_cells):
			ansRow = []
			for v in range(0, v_cells-(1 if self.vlabel is None else 2)):
				i = u - (1 if self.hlabel is None else 2); j = v
				x = int(cell_w*v); y = int(cell_h*u); w = int(cell_w); h = int(cell_h)
				offset_l = int(min(cell_w*offset, x)); offset_r = int(min(cell_w*offset, width-x)); offset_u = int(min(cell_h*offset, y)); offset_d = int(min(cell_h*offset, height-y))
				xo = x-offset_l; yo = y-offset_u; wo = w+offset_l+offset_r; ho = h+offset_u+offset_d
				roi = img[yo:yo+ho,xo:xo+wo]

				successful = False
				for ci in ImageUtils.markerDetector(roi, minHierarchy=1, smallArea=True):
					c = (int(ci[0][0]), int(ci[0][1]), int(ci[1]))
					if c[2] < rad_min or c[2] > rad_max:
						continue
					border_h = cell_h * border; border_w = cell_w * border
					if c[1]-offset_u > border_h and c[1]-offset_u < cell_h-border_h and c[0]-offset_l > border_w and c[0]-offset_l < cell_w-border_w:
						roi_gray = img_gray[yo:yo+ho,xo:xo+wo]
						circle_mask = np.zeros((ho, wo), np.uint8)
						cv2.circle(circle_mask,(c[0],c[1]),c[2],255,-1)
						avg = cv2.mean(roi_gray, mask=circle_mask)[::-1][-1]
						ansRow.append({'checked': avg>200, 'center': (xo+c[0], yo+c[1]), 'radius': c[2], 'template': template[i][j]})
						successful = True
						break
				if not successful:
					all_successful = False
					if __debug__:
						ansRow.append({'error': True, 'x0': xo, 'y0': yo, 'x1': xo+wo-1, 'y1': yo+ho-1})
					else:
						break
			ans.append(ansRow)
			if not all_successful and not __debug__:
				break

		if __debug__:
			for ans_row in ans:
				for e in ans_row:
					if 'error' in e:
						cv2.line(img, (e['x0'],e['y0']), (e['x1'],e['y1']), (0,0,125), thickness=1, lineType=cv2.LINE_AA)
						cv2.line(img, (e['x0'],e['y1']), (e['x1'],e['y0']), (0,0,125), thickness=1, lineType=cv2.LINE_AA)
					else:
						cv2.circle(img, e["center"], e["radius"], (255,0,0), thickness=2, lineType=cv2.LINE_AA)

			import sys
			img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
			img_gray = np.vstack((img, img_gray))
			cv2.imshow(__class__.__name__ + "." + sys._getframe(1).f_code.co_name, img_gray)

		# If not possible to read all answers, abort!
		if not all_successful:
			return None, None, None

		# Make answer matrix to check...
		matrix = []
		for ans_row in ans:
			matrix_row = []
			for e in ans_row:
				matrix_row.append(e['checked'])
			matrix.append(matrix_row)

		# Calcuate the questions score
		score = self.getScore(matrix)

		# Draw wrongs and corrects answers to show feedback
		for ar in ans:
			for a in ar:
				if a['checked'] and a['template']:
					cv2.circle(img, a['center'], a['radius'], (0,255,0), -1)
				elif a['checked'] and not a['template']:
					cv2.circle(img, a['center'], a['radius'], (0,0,255), -1)
				elif not a['checked'] and a['template']:
					cv2.circle(img, a['center'], a['radius']//2, (0,255,0), -1)
					cv2.circle(img, a['center'], a['radius'], (0,0,255), 3)
				else:
					cv2.circle(img, a['center'], a['radius'], (0,255,0), 2)

		# Draw grid lines
		for i in range(h_cells):
			cv2.line(img, (0,int(cell_h*i)), (int(width), int(cell_h*i+1)), (0,255,0))
		for j in range(v_cells):
			cv2.line(img, (int(cell_w*j), 0), (int(cell_w*j), int(height)), (0,255,0))

		# Make Feedback Image
		imgInfo = np.zeros((60, img.shape[1], 3), np.uint8)
		imgInfo[:,:] = (255,255,255)
		ImageUtils.drawTextInsideTheBox(imgInfo, self.getAnswerText(LaTeX=False))
		imgInfo = np.vstack((imgInfo, img))

		return score, img, imgInfo
# END QUESTION MATRIX #
#######################




########################
# BEGIN QUESTION ESSAY #
class QuestionEssay(QuestionMatrix):
	def makeVariables(self):
		raise Exception("Method not implemented.")
	def getQuestionTex(self):
		raise Exception("Method not implemented.")
	def getAnswerText(self, LaTeX):
		raise Exception("Method not implemented.")
		
	def setLabels(self, scores, blank_warning):
		self.cols = scores
		self.rows = [""]
		self.hlabel = blank_warning
		self.vlabel = None
	def answerAreaAspectRate(self):
		return 10/1
	def getScore(self, matrix_answer):
		for i in range(len(self.cols)):
			if matrix_answer[0][i]:
				return self.cols[i]
	def getTemplate(self):
		return [[False for j in range(len(self.cols))] for i in range(len(self.rows))]
# END QUESTION ESSAY #
######################



#############################
# BEGIN QUESTION TRUE FALSE #
class QuestionTrueOrFalse(QuestionMatrix):
	questions            = None
	questionDescription  = None
	scoreTable           = None
	wrongFactor          = None
	labels               = None
	conditionByLineLabel = None

	def makeSetup(self):
		raise Exception("Method not implemented.")

	def answerAreaAspectRate(self):
		return 10/1

	def makeVariables(self):
		self.makeSetup()
		self.rows   = [self.labels["true"], self.labels["false"]]
		self.cols   = [str(i) for i in range(1,1+len(self.questions))]

	def getScore(self, matrix_answer):
		correct = 0; wrong = 0; blank = 0
		for i in range(len(self.questions)):
			if    matrix_answer[0][i] == True  and matrix_answer[1][i] == True:  wrong   += 1
			elif  matrix_answer[0][i] == False and matrix_answer[1][i] == False: blank   += 1
			elif self.questions[i][1] == True  and matrix_answer[0][i] == True:  correct += 1
			elif self.questions[i][1] == False and matrix_answer[1][i] == True:  correct += 1
			else:                                                                wrong   += 1

		if self.wrongFactor == 0:
			sum = correct
		else:
			from math import ceil
			sum = ceil(correct - (wrong / self.wrongFactor))

		self.scoreTable.sort(key=lambda x: x[0], reverse=True)
		for s in self.scoreTable:
			if s[0] <= sum:
				return s[1]
		return self.scoreTable[-1][1]

	def getTemplate(self):
		rowT = [True  == q[1] for q in self.questions]
		rowF = [False == q[1] for q in self.questions]
		return [rowT, rowF]

	def getAnswerText(self, LaTeX):
		ret = ""
		for q in range(len(self.questions)):
			if LaTeX:
				ret += "\\tiny{{{num}}}\\normalsize{{{ans}}} ".format(num=q+1, ans= self.labels["true"] if self.questions[q][1] else self.labels["false"])
			else:
				ret += "{num}:{ans} ".format(num=q+1, ans= self.labels["true"] if self.questions[q][1] else self.labels["false"])			
		return ret

	def getQuestionTex(self):
		tex = self.questionDescription

		tex += "\\begin{enumerate}[topsep=0pt,itemsep=-1ex,partopsep=1ex,parsep=1ex, label=\\textbf{\\arabic*.}]\n"
		for q in self.questions:
			tex += "\\item {quest}\n\n".format(quest=q[0])
		tex += "\\end{enumerate}"

		# Print correction criteria
		tex += "\n$\n\\begin{cases}\n"
		if self.wrongFactor == 0:
			tex += "X = \\text{{{t}}} \\\\".format(t=self.labels['number_of_right_questions'])
		else:
			tex += "X = \\lceil \\text{{{t}}} - \\frac {{\\text{{{f}}}}} {{{wf}}} \\rceil \\\\".format(max=len(self.questions),t=self.labels['number_of_right_questions'], f=self.labels['number_of_wrong_questions'], wf=self.wrongFactor)
		tex += "\\text{{{}}} = \\begin{{cases}}".format(self.labels['score'])
		count = 0
		for s in sorted(self.scoreTable, key=lambda x: x[0], reverse=True):
			if s[0] == 0:
				tex += "\\text{{ {} }} {}".format(self.labels['else'], s[1])
				break
			tex += " {}\\text{{ {} }}X>={}".format(s[1], self.labels['if'], s[0])
			if count % self.conditionByLineLabel == self.conditionByLineLabel-1 and count != len(self.scoreTable) - 1:
				tex += "\\\\"
			else:
				tex += "\\text{; }"
			count += 1
		tex += "\\end{cases}"
		tex += "\n\\end{cases}\n$\n"

		return tex
# END QUESTION TRUE FALSE #
###########################



##################################
# BEGIN QUESTION MULTIPLE CHOICE #
class QuestionMultipleChoice(QuestionMatrix):
	questions           = None
	scoreTable          = None
	questionDescription = None
	labels              = None
	conditionByLineLabel = None

	def makeSetup(self):
		raise Exception("Method not implemented.")

	def answerAreaAspectRate(self):
		return (32)/(len(self.questions[0]['alternatives']) + (1 if self.hlabel is None else 2))

	def makeVariables(self):
		self.makeSetup()
		num_alternative = len(self.questions[0]['alternatives'])
		for i in self.questions:
			if len(i['alternatives']) != num_alternative:
				raise Exception("Inconsistent choice quantity: {}".format(self.questions))

		self.rows   = [chr(ord("A") + i) for i in range(num_alternative)]
		self.cols   = [str(i) for i in range(1,1+len(self.questions))]

	def getScore(self, matrix_answer):
		corrects = 0
		for j in range(len(matrix_answer[0])):
			correct = True
			for i in range(len(matrix_answer)):
				if matrix_answer[i][j] != self.questions[j]['alternatives'][i][1]:
					correct = False
					break
			if correct:
				corrects += 1
		self.scoreTable.sort(key=lambda x: x[0], reverse=True)
		for s in self.scoreTable:
			if s[0] <= corrects:
				return s[1]
		return self.scoreTable[-1][1]

	def getTemplate(self):
		matrix = [[None for j in range(len(self.questions))] for i in range(len(self.questions[0]['alternatives']))]
		for i in range(len(self.questions)):
			for j in range(len(self.questions[i]['alternatives'])):
				matrix[j][i] = self.questions[i]['alternatives'][j][1]
		return matrix

	def getAnswerText(self,LaTeX):
		ret = ""
		for q in range(len(self.questions)):
			for a in range(len(self.questions[q]['alternatives'])):
				if self.questions[q]['alternatives'][a][1]:
					if LaTeX:
						ret += "\\tiny{{{num}}}\\normalsize{{{ans}}} ".format(num=q+1, ans=str(chr(ord("A")+a)))
					else:
						ret += "{num}: {ans} ".format(num=q+1, ans=str(chr(ord("A")+a)))
					break
		return ret

	def getQuestionTex(self):
		tex = self.questionDescription
		tex += "\n\n"
		tex += "\\begin{enumerate}[topsep=0pt,itemsep=-1ex,partopsep=1ex,parsep=1ex, label=\\textbf{\\arabic*.}]\n"
		for q in self.questions:
			tex += "\\item {quest}\n\n".format(quest=q['statement'])
			if "itensPerRow" in q and q["itensPerRow"] > 1:
				tex += "\\begin{{multicols}}{{{}}}".format(q["itensPerRow"])
			tex += "\\begin{enumerate}[topsep=-2ex,itemsep=-1ex,partopsep=1ex,parsep=1ex, label=\\textbf{\\Alph*)}]"
			for i in q['alternatives']:
				tex += "\\item {alter}\n\n".format(alter=i[0])
			tex += "\\end{enumerate}"
			if "itensPerRow" in q and q["itensPerRow"] > 1:
				tex += "\\end{multicols}"
		tex += "\end{enumerate}"

		# Print correction criteria
		tex += "\n\n$\n\\begin{cases}\n"
		tex += "X = \\text{{{t}}} \\\\".format(t=self.labels['number_of_right_questions'])
		tex += "\\text{{{}}} = \\begin{{cases}}".format(self.labels['score'])
		count = 0
		for s in sorted(self.scoreTable, key=lambda x: x[0], reverse=True):
			if s[0] == 0:
				tex += "\\text{{ {} }} {}".format(self.labels['else'], s[1])
				break
			tex += " {}\\text{{ {} }}X>={}".format(s[1], self.labels['if'], s[0])
			if count % self.conditionByLineLabel == self.conditionByLineLabel-1 and count != len(self.scoreTable) - 1:
				tex += "\\\\"
			else:
				tex += "\\text{; }"
			count += 1
		tex += "\\end{cases}"
		tex += "\n\\end{cases}\n$\n"

		return tex
# END QUESTION MULTIPLE CHOICE #
################################



#########################
# BEGIN QUESTION NUMBER #
class QuestionNumber(QuestionMatrix):
	max_digits        = None
	decimal_separator = None
	expected_value    = None

	def getScoreFromNumber(self, num):
		raise Exception("Method not implemented.")
	def makeSetup(self):
		raise Exception("Method not implemented.")

	def answerAreaAspectRate(self):
		return (28)/(self.max_digits + (1 if self.hlabel is None else 2))

	def makeVariables(self):
		self.makeSetup()
		self.rows = [str(i) for i in range(1,1+self.max_digits)]
		self.cols = [str(i) for i in range(0,10)]
		if self.decimal_separator is not None:
			self.cols.append(self.decimal_separator)

	def getScore(self, matrix_answer):
		digits = []
		for d in range(len(matrix_answer)):
			num = None
			for n in range(len(matrix_answer[d])):
				if matrix_answer[d][n]:
					if num is None: num = n
					else:           return self.getScoreFromNumber(None) # Invalid
			digits.append(num)

		dig = ""
		for d in digits:
			if d is None: continue
			else:         dig += str(d) if d <= 9 else "."

		try: value = float(dig)
		except ValueError: return self.getScoreFromNumber(None) # Invalid
		return self.getScoreFromNumber(value)

	def getTemplate(self):
		matrix = [[False for j in range(len(self.cols))] for i in range(len(self.rows))]
		str_val = str(self.expected_value)
		for d in range(len(str_val)):
			if str_val[d] == ".":
				matrix[d][10] = True
			else:
				matrix[d][int(str_val[d])] = True
		return matrix
# END QUESTION NUMBER #
#######################



######################################
# BEGIN QUESTION QUESTION AND ANSWER #
class QuestionQA(QuestionMatrix):
	full_list_question_answer_text = None
	list_question_answer_index     = None
	question_description           = None
	scoreTable                     = None
	labels                         = None
	conditionByLineLabel           = None

	@staticmethod
	def auxShuffle(questions, num_questions):
		import random
		from copy import deepcopy
		quest = list(range(0,len(questions)))
		random.shuffle(quest)
		quest = quest[0:num_questions]
		ans = deepcopy(quest)
		random.shuffle(ans)
		return [(quest[i], ans[i]) for i in range(0,num_questions)]

	def makeSetup(self):
		raise Exception("Method not implemented.")

	def answerAreaAspectRate(self):
		return (28)/(len(self.list_question_answer_index) + (1 if self.hlabel is None else 2))

	def makeVariables(self):
		self.makeSetup()
		self.rows = [chr(ord("A") + i) for i in range(len(self.list_question_answer_index))]
		self.cols = [str(i) for i in range(1,1+len(self.list_question_answer_index))]
		self.hlabel = self.labels["questions"]
		self.vlabel = self.labels["answers"]

	def getQuestionTex(self):
		tex = self.question_description + "\n"
		tex += "\n\\begin{tabularx}{\\textwidth}{|c|c||c|X|}\n\\hline\n"
		tex += "\\multicolumn{{2}}{{|c||}}{{\\textbf{{{q}}}}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{{a}}}}} \\\\\n".format(q=self.labels['questions'], a=self.labels['answers'])

		for q in range(0,len(self.list_question_answer_index)):
			tex += "\\hline\n\\textbf{{{n1}}} & {p} & \\textbf{{{n2}}} & {r} \\\\\n".format( p=self.full_list_question_answer_text[self.list_question_answer_index[q][0]][0],
            r=self.full_list_question_answer_text[self.list_question_answer_index[q][1]][1],
            n1=str(q+1), n2=chr(ord('A') + q)+":")
		tex += "\\hline\n\end{tabularx}\n"

		# Print correction criteria
		tex += "\n\n$\n\\begin{cases}\n"
		tex += "X = \\text{{{t}}} \\\\".format(t=self.labels['number_of_right_questions'])
		tex += "\\text{{{}}} = \\begin{{cases}}".format(self.labels['score'])
		count = 0
		for s in sorted(self.scoreTable, key=lambda x: x[0], reverse=True):
			if s[0] == 0:
				tex += "\\text{{ {} }} {}".format(self.labels['else'], s[1])
				break
			tex += " {}\\text{{ {} }}X>={}".format(s[1], self.labels['if'], s[0])
			if count % self.conditionByLineLabel == self.conditionByLineLabel-1 and count != len(self.scoreTable) - 1:
				tex += "\\\\"
			else:
				tex += "\\text{; }"
			count += 1
		tex += "\\end{cases}"
		tex += "\n\\end{cases}\n$\n"

		return tex

	def getAnswerText(self,LaTeX):
		q = [q for q, a in self.list_question_answer_index]
		a = [a for q, a in self.list_question_answer_index]
		ans = ""
		for n in range(len(self.list_question_answer_index)):
			if LaTeX:
				ans += "\\tiny{{{n}}}\\normalsize{{{r}}} ".format(r=chr(ord('A') + a.index(q[n])), n=n+1)
			else:
				ans += "{n}: {r} ".format(r=chr(ord('A') + a.index(q[n])), n=n+1)
		return ans

	def getScore(self, matrix_answer):
		template = self.getTemplate()
		corrects =0; wrongs = 0
		for i in range(len(matrix_answer)):
			for j in range(len(matrix_answer[i])):
				if template[i][j]:
					valid = True
					for s in range(len(matrix_answer)):
						if matrix_answer[s][j] != template[s][j]:
							valid = False
							break
					if valid == False:
						break
					for t in range(len(matrix_answer[i])):
						if matrix_answer[i][t] != template[i][t]:
							valid = False
							break
					if valid:
						corrects += 1

		self.scoreTable.sort(key=lambda x: x[0], reverse=True)
		for s in self.scoreTable:
			if s[0] <= corrects:
				return s[1]
		return self.scoreTable[-1][1]

	def getTemplate(self):
		matrix = [[False for j in range(len(self.cols))] for i in range(len(self.rows))]
		q = [q for q, a in self.list_question_answer_index]
		a = [a for q, a in self.list_question_answer_index]
		for i in range(len(q)):
			matrix[i][a.index(q[i])] = True
		return matrix
# END QUESTION QUESTION AND ANSWER #
####################################



######################
# BEGIN QUESTION OCR #
class QuestionOCR(Question):
	def getScore(self, text):
		raise Exception("Method not implemented.")
	def drawAnswerArea(self, img):
		import cv2
		return cv2.rectangle(img, (0,0), (img.shape[1],img.shape[0]),(0,0,0), thickness=2, lineType=cv2.LINE_AA)
	def doCorrection(self, img):
		import pytesseract
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.medianBlur(img_gray, 5)
		img_gray = cv2.GaussianBlur(img_gray, (7,7),0)
		img_gray = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,C=2)
		img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
		return self.getScore(pytesseract.image_to_string(img_gray)), img_gray
# END QUESTION OCR #
####################



#####################
# BEGIN QUESTIONSDB #
class QuestionsDB:
	salt      = None
	questions = None

	def __init__(self, base_path, salt=""):
		self.questions = Utils.loadModules(base_path)
		self.salt      = salt

	def getQuestions(self, selected_groups, uniqueid=None, **args_to_question):
		import random
		if uniqueid is None:
			uniqueid = self.generateUniqueId(**args_to_question)

		random.seed(uniqueid)

		# Create and shuffle questions from each group/subgroup
		listQuest = dict()
		from types import ModuleType
		for grp in selected_groups:
			quest = self.questions
			for g in grp.split('/'):
				if not g in quest:
					raise Exception("Group '{}' didn't exist".format(grp))				
				quest = quest[g]
			if type(quest) is ModuleType: # If the path is a Python module (not a directory)
				listQuest[grp] = [[None, quest]]
			else:
				quests = [(k,v) for k,v in quest.items() if type(v) is ModuleType]
				random.shuffle(quests) # Shuffle questions
				listQuest[grp] = quests

		result = []
		for grp in selected_groups:
			if len(listQuest[grp]) == 0:
				raise Exception("There is no more questions in '{}'".format(grp))
			question = QuestionsDB.instantiateQuestion(listQuest[grp][0][1], **args_to_question)
			if question is not None:
				random.seed(uniqueid)
				question.makeVariables() #Shuffle variables
				result.append(question)
				listQuest[grp].pop(0)
		return result, uniqueid

	@staticmethod
	def instantiateQuestion(module, **args):
		import inspect
		for name, obj in inspect.getmembers(module):
			if inspect.isclass(obj) and obj.__module__ == module.__name__:
				for b in inspect.getmro(obj):
					if b.__name__ == Question.__name__: # TODO Best way??
						return obj(**args)
		print("Invalid module")
		return None

	def generateUniqueId(self, **args_to_question):
			uniqueid = 0
			for k,v in args_to_question.items():
				uniqueid += Utils.str2intHash(str(k)+str(v)+self.salt)
			return uniqueid
# END QUESTIONSDB #
###################



######################
# CORRECTION MANAGER #
class CorrectionManager:
	full_path        = None # Path to all data correction directory
	csv_param        = None # Parameters used at csv Writer and Reader (delimiterand quotechar)
	csv_file         = None # Filename of csv with scores information, inside of correction directory
	students         = None # List of students
	scoreTable       = None # Table with all students and scores. This table is equal of csv file
	headerLabels     = None # Structure type dict with header fields, used in csv file
	fieldnames       = None # Header labels of csv file
	fieldnames_id    = None
	fieldsname_quest = None
	fieldnames_final = None
	num_quest        = None # Number of questions
	weight           = None # List with weight of each question
	final_calc_str   = None # Algorithmn used to calculate the final score

	def __init__(self, students, weight, path, csv_file, delimiter=" ", quotechar="\"", final_calc=None, headers=None):
		import os, collections, copy
		self.full_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), path)
		if not os.path.exists(self.full_path):
			os.makedirs(self.full_path)

		self.students         = students
		self.csv_file         = self.full_path + "/" + csv_file
		self.csv_param        = {"delimiter": delimiter, "quotechar": quotechar}
		self.num_quest        = len(weight)
		self.weight           = weight
		self.headerLabels     = headers
		self.fieldnames_id    = []
		self.fieldsname_quest = []
		self.fieldnames_final = []

		# Get final calculation algorithmn
		self.final_calc_str = ""
		for f in final_calc: self.final_calc_str += f + "\n"

		# Get field names
		self.fieldnames_id = [v for k,v in self.headerLabels['identification'].items()]
		for i in range(self.num_quest):
			self.fieldsname_quest.append(self.headerLabels['intermediate'].replace(self.headerLabels['counter'], str(i+1)))
		self.fieldnames_final.append(self.headerLabels['final'])
		self.fieldnames = self.fieldnames_id + self.fieldsname_quest + self.fieldnames_final
		
		# Make an empty score table with all students
		self.scoreTable = []
		for s in self.students:
			line = collections.OrderedDict()
			for k,v in self.headerLabels['identification'].items():
				if k in s:
					line[v] = s[k]
			self.scoreTable.append(line)
			for i in range(self.num_quest):
				line[self.headerLabels['intermediate'].replace(self.headerLabels['counter'], str(i+1))] = None
			line[self.headerLabels['final']] = None

		# Load saved scores
		self.load()
		self.save()
	
	def final_calc(self, questions_score):
		exec(self.final_calc_str)
		return eval('final_calc(questions_score, self.weight)')

	def load(self):
		import csv
		try:
			import csv, collections
			with open(self.csv_file, 'r') as f:
				reader = csv.DictReader(f, **self.csv_param)
				for line in reader:
					l = collections.OrderedDict(sorted(line.items(), key=lambda item: reader.fieldnames.index(item[0])))
					self.updateScores(l)
				f.close()
		except csv.Error as e:
			raise Exception("{} ({}:{})".format(e, self.csv_file, reader.line_num))
		except FileNotFoundError:
			return # Ok, no problems...
		except Exception as e:
			self.save() # Error to read csv data. Overwrite with new one.

	def save(self):
		import csv
		with open(self.csv_file, 'w') as f:
			writer = csv.DictWriter(f, fieldnames=self.fieldnames, **self.csv_param)
			writer.writeheader()
			writer.writerows(self.scoreTable)
			f.close()

	def getIdentification(self, student):
		ret = dict()
		for k,v in self.headerLabels['identification'].items():
			if v in student: # If student is from scoreTable
				ret[v] = student[v]
			elif k in student: # If student is from input.csv
				ret[v] = student[k]
		return ret

	def findStudent(self, student):
		for s in self.scoreTable:
			found = True
			for k,v in self.headerLabels['identification'].items():
				if not((v in student and s[v] == student[v]) or (k in student and s[v] == student[k])):
					found = False
					break
			if found:
				return s
		return None

	def getScores(self, student):
		return self.findStudent(self.getIdentification(student))

	def updateScores(self, line_dict):
		student = self.findStudent(self.getIdentification(line_dict))

		# If student don't exist, insert it and try again
		if student is None:
			self.scoreTable.append(line_dict)
			return self.updateScores(line_dict)
	
		# Get all updates... just modified values!
		from collections import OrderedDict
		ret = OrderedDict() # List of all modified values
		all_questions = []
		changed = False
		for i in range(self.num_quest):
			f = self.headerLabels['intermediate'].replace(self.headerLabels['counter'], str(i+1))
			if line_dict[f] is not None and line_dict[f] != '':
				if student[f] != line_dict[f]:
					student[f] = line_dict[f]
					ret[f] = line_dict[f]
					changed = True
			if student[f] is not None and student[f] != '':
				all_questions.append(student[f])

		# Calculate/recalculate final score (if necessary or possible)
		if len(all_questions)==self.num_quest and (changed or student[self.headerLabels['final']] is None or student[self.headerLabels['final']] == ''):
			student[self.headerLabels['final']] = self.final_calc(all_questions)
			ret[self.headerLabels['final']] = student[self.headerLabels['final']]

		# Return the list with all modified or calculated scores
		return ret, student

	def updateScore(self, student, question_num, score):
		question_field = self.headerLabels['intermediate'].replace(self.headerLabels['counter'], str(question_num+1))

		# Create a line (s) with identifications and scores of a student to update later (in return)
		from collections import OrderedDict
		s = OrderedDict()
		for f in self.fieldnames:
			# student identification from input.csv has a different header of scoreTable. Finding the corresponding field header
			fk = f
			for k,v in self.headerLabels['identification'].items():
				if v == f:
					fk = k

			# Copy student identification
			if fk in student:
				s[f] = student[fk]

			# Update a specific question field
			elif f == question_field:
				s[f] = score

			# Leave others fields with unkown values blank
			else:
				s[f] = None

		# Update all scores and return
		return self.updateScores(s)
# CORRECTION MANAGER #
######################



########
# MAIN #
class Main:
	config       = None # Data Structure with JSON Config
	questionsDB  = None # Questions Data Base
	students     = None # List of Students
	verbose      = None # Verbose, from 0 to 3
	selected     = None # Selected questions group
	correction   = None # Correction Manager Class, used to save corrections
	temp_dir     = None # If not None, specifies a temporary directory and don't delete it.

	def __init__(self, config_file=None, config_default=None, verbose=0, temp_dir=None):
		self.verbose = verbose
		self.temp_dir = temp_dir

		# READ CONFIG.JSON
		if config_default is not None:
			self.config = Utils.json2dict(config_default)
		else:
			from collections import OrderedDict
			self.config = OrderedDict()
		with open(config_file) as f:
			import os
			os.chdir(os.path.dirname(os.path.realpath(config_file))) # All path will be relative to the config file...
			Utils.jsonMerge(self.config, Utils.json2dict(f.read()), raiseErrorFromUnexpected=True)
			f.close()
		# END CONFIG.JSON

		try: # READ STUDENTS.CSV
			import csv
			self.students = []
			encode = Utils.getEncodeFile(self.config['input']['filename'])
			with open(self.config['input']['filename'], 'r', encoding=encode) as f:
				from collections import OrderedDict
				reader = csv.DictReader(f, delimiter=self.config['input']['delimiter'], quotechar=self.config['input']['quotechar'])
				for line in reader:
					s = OrderedDict(sorted(line.items(), key=lambda item: reader.fieldnames.index(item[0])))
					self.students.append(s)
				f.close()
		except csv.Error as e:
			raise Exception("Parser error in {}:{}: {}".format(e, self.config['input']['filename'], reader.line_num))
		except FileNotFoundError:
			raise Exception("CSV file '{}' not found!".format(self.config['input']['filename']))
		except Exception as e:
			raise Exception("Error in '{}': {}".format(self.config['input']['filename'], e))
		#END CSV

		# Instantiate CorrectionManager class
		self.correction = CorrectionManager(students=self.students, weight=[i['weight'] for i in self.config['questions']['select']], **self.config['correction'])

		# Load Questions' Repository and list of selected question's path
		self.questionsDB = QuestionsDB(self.config['questions']['db_path'], salt=self.config['questions']['salt'])
		self.selected    = [c['path'] for c in self.config['questions']['select']]

	def makeDirectoryForAStrudent(self, student):
		import os

		# Make path to correction directory
		full_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), self.config['correction']['path'])
		if not os.path.exists(full_path): os.makedirs(full_path)

		# Make an unique directory name for each student using student_directory_id info.
		student_str = self.config['correction']['headers']['student_directory_id']
		for k,v in self.config['correction']['headers']['identification'].items():
			student_str = student_str.replace(k,str(student[k]))
		
		# Make path to student directory
		full_path_student = full_path + "/" + student_str
		if not os.path.exists(full_path_student): os.makedirs(full_path_student)
		
		return full_path, full_path_student

	def findStudentFromAllQRCode(self, img):
		# Find all valid QRCode
		for countour, data in ImageUtils.qrCodeDecoder(img):
			import ast, collections
			try:
				# Workaround for special characters (https://sourceforge.net/p/zbar/discussion/664596/thread/ed7aca9d/)
				data = ast.literal_eval(data.decode("utf-8").encode("sjis").decode('utf-8'))

			except SyntaxError: continue # Invalid data
			except ValueError:  continue # Invalid data
			if type(data) is list and len(data) == 3:
				qrcode_page    = int(data[0])
				qrcode_idx     = int(data[1])
				qrcode_student = collections.OrderedDict(data[2])
				if qrcode_page < 1 or qrcode_idx < 0 or len(qrcode_student) == 0:
					continue # Invlalid data

				if self.verbose > 2:
					print("QRCode found: page {}, student {}".format(qrcode_page, qrcode_student), end="")
					for k,v in qrcode_student.items():
						print("{}: {} ".format(k,v), end="")
					print(".")

				# Trying to get the student info using the index readed from QRCode
				found = True
				if qrcode_idx >= len(self.students):
					break
				for k,v in qrcode_student.items():
					if k not in self.students[qrcode_idx] or self.students[qrcode_idx] != v:
						found = False
						break
				if found:
					qrcode_valid_student = self.students[qrcode_idx]

				# If the index is wrong, iterate all students to find it
				elif not found:
					for s in self.students:
						found = True
						for k,v in qrcode_student.items():
							if k not in s or s[k] != v:
								found = False
								break
						if found:
							qrcode_valid_student = s
							break

				# If the current student found, save the current page image into your correction directory
				if found:
					yield qrcode_valid_student, qrcode_page, countour

	def makeTests(self):
		# Initiate output tests
		tex = LaTeX(sufix_temp_dir="Tests", temporary_directory=self.temp_dir)
		tex.addReplaces({self.config['tex']['question_total']: str(len(self.selected))})
		tex.addReplaces(self.config['tex']['replaces'])
		for i in self.config['tex']['includes']:
			tex.addInclude(i)
		tex.addTex(self.config['tex']['preamble'])

		#Initiate output template
		template = LaTeX(sufix_temp_dir="Template", temporary_directory=self.temp_dir)
		template.addReplaces({self.config['tex']['question_total']: str(len(self.selected))})
		template.addReplaces(self.config['tex']['replaces'])
		for i in self.config['tex']['includes']:
			template.addInclude(i)
		template.addTex(self.config['tex']['preamble'])
		template.addTex(self.config['tex']['template']['header'])

		if self.verbose > 2:
			print("Temporary Directory (Tests):    {}".format(tex.temp_dir))
			print("Temporary Directory (Template): {}".format(template.temp_dir))

		if self.verbose > 1: print("There is {} students:".format(len(self.students)))

		# Create test for each student
		for student_idx in range(len(self.students)):
			student = self.students[student_idx]

			if self.verbose > 1:
				print("[{:.0f}%]".format(100*student_idx/len(self.students)), end="")
				for k,v in student.items():
					print("\t{}: {}".format(k,v), end="")
				print(".")
			elif self.verbose == 1:
				import sys
				print("\rThere is {} students:\t[{:.0f}%]".format(len(self.students), 100*student_idx/len(self.students)), end="")
				sys.stdout.flush()

			# Get questions for a specifi student s
			questions, uniqueid = self.questionsDB.getQuestions(self.selected, uniqueid=None, **student)
			if self.verbose > 2: print("\t\tLoaded {} questions:".format(len(questions)))

			# Insert tex.addImage method into question
			for q in questions:
				q.addImage = tex.addImage

			# Update student info in LaTeX macro
			tex.addReplaces(student)
			template.addReplaces(student)

			# Make QRCode (one by page) and save it in latex temporary folder
			for pg in range(1,self.config['tex']['max_pages']+1):
				data = [pg, student_idx, dict(student)]
				img = ImageUtils.makeQRCode(str(data))
				sufix='-' + str(pg); extension=".png"
				qrcodeid_img_name = tex.addImage(img, prefix="IfThisFileNotFound-Check-ConfigJson-Tex-MaxPages-", sufix=sufix, extension=extension)
				tex.addReplaces({self.config['tex']['qrcode_id_must_be_concatenated_with_dash_plus_the_page_number']: qrcodeid_img_name[:len(qrcodeid_img_name)-len(sufix)-len(extension)]})

			# Begin new student
			tex.addTex(self.config['tex']['test']['header'])
			template.addTex(self.config['tex']['template']['before'])
			
			# Insert questions in tex
			for i in range(len(questions)):
				if self.verbose > 2: print("\t\t\t{}".format(questions[i].__class__.__name__))

				# Generate an unique code for each question
				code = str(uniqueid)+"-"+str(student_idx)+"-"+str(i)
				img, ans_img = ImageUtils.makeAnswerArea(code, questions[i].answerAreaAspectRate())
				questions[i].drawAnswerArea(ans_img)
				ans_area_image_path = tex.addImage(img,prefix="AnswerArea-", extension=".png")
				tex.addReplaces({self.config['tex']['question_image_answer_area']: ans_area_image_path})

				# Get the correct answer
				ans_text = questions[i].getAnswerText(LaTeX=True)

				# Insert question description in tex
				tex.addReplaces({self.config['tex']['question_counter']: str(i+1)})
				tex.addReplaces(self.config['questions']['select'][i]['replaces'])
				tex.addReplaces({self.config['tex']['answer_text']: ans_text})
				tex.addTex(self.config['tex']['test']['before'])
				tex.addTex(questions[i].getQuestionTex())
				tex.addTex(self.config['tex']['test']['after'])

				# Insert correct answer in template
				template.addReplaces({self.config['tex']['question_counter']: str(i+1)})
				template.addReplaces({self.config['tex']['answer_text']: ans_text})
				template.addTex(self.config['tex']['template']['answer'])

			# End new student
			tex.addTex(self.config['tex']['test']['footer'])
			template.addTex(self.config['tex']['template']['after'])

		# Finisth tex and template document
		tex.addTex(self.config['tex']['termination'])
		template.addTex(self.config['tex']['template']['footer'])
		template.addTex(self.config['tex']['termination'])

		from subprocess import CalledProcessError
		try:
			# Make tests.pdf
			if self.verbose == 1:
				print("\rThere is {} students:\t[100%]".format(len(self.students))) # Overwrite percents
			if self.verbose > 2:
				tex.printTex()
			if self.verbose > 0:
				print("Generating {}...".format(self.config['output']['tests']))
			for l in tex.makePdf(self.config['output']['tests']):
				if self.verbose > 2:
					print(l, end="")
			if self.verbose > 0:
				print("PDF '{}' generated.".format(self.config['output']['tests']))
		except CalledProcessError as e:
			if   self.verbose > 2: raise Exception(tex.getError() + "\n\noutput.log:\n===========\n\t" + e.output.replace("\n", "\n\t"))
			elif self.verbose > 1: raise Exception(tex.getError())
			else:                  raise Exception("Error to genereate {}". format(self.config['output']['tests']))
		except UnicodeDecodeError as e:
			if self.verbose > 1: raise Exception(e.args)
			else:                raise Exception("Error to genereate {}". format(self.config['output']['tests']))


		# Make template.pdf
		try:
			if self.verbose > 2:
				template.printTex()
			if self.verbose > 0:
				print("Generating {}...".format(self.config['output']['template']))
			for l in template.makePdf(self.config['output']['template']):
				if self.verbose > 2:
					print(l, end="")
			if self.verbose > 0:
				print("PDF '{}' generated.".format(self.config['output']['template']))
		except CalledProcessError as e:
			if self.verbose > 1: raise Exception(template.getError())
			else:                raise Exception("Error to generate {}". format(self.config['output']['template']))
		except UnicodeDecodeError as e:
			if self.verbose > 1: raise Exception(e.args)
			else:                raise Exception("Error to genereate {}". format(self.config['output']['template']))

	def doCorrection(self, img):
		# Resize for large image
		img = ImageUtils.conditionalResize(img, max_res=ImageUtils.MAX_IMAGE_RES)

		# Save the original image to copy fully in correction folder
		img_orig = img.copy()

		# Find all QRCode with student's information and save the current image in the student directory
		qrcode_students = []
		for s, p, c in self.findStudentFromAllQRCode(img):
			# Save the current page image into student correction directory
			full_path, full_path_student = self.makeDirectoryForAStrudent(s)
			if self.verbose > 0:
				print("Save page {} into directory of student ".format(p), end="")
				for k,v in s.items():
					print("{}: {} ".format(k,v), end="")
				print(".")
				cv2.drawContours(img, [c], 0, (0,0,255), thickness=2)
			cv2.imwrite(full_path_student + "/{}.jpg".format(p), img_orig)
			qrcode_students.append(s)

		# Find all answer area in current image
		for ansArea, code, contours, markers in ImageUtils.findAnswerAreas(img, verbose=self.verbose>0):
			# Get unique_id, student_idx and question_num from answer area
			unique_id, student_idx, question_num = code.split("-")
			unique_id = int(unique_id); student_idx = int(student_idx); question_num = int(question_num)
			if self.verbose > 2:
				print("Answer area found with student_index = {}, question_number = {} and unique_id = {}".format(student_idx, question_num, unique_id))

			# Get student info and questions
			if student_idx >= len(self.students):
				continue
			student = self.students[student_idx]
			questions, unique_id_calculated = self.questionsDB.getQuestions(self.selected, uniqueid=None, **student)

			# Check if the student of current answer area is the same of found in student list with specific index
			if (unique_id != unique_id_calculated):
				if self.verbose > 1:
					print("Warning: {} file is inconsistent with current correction! Trying to scan QRCode.".format(self.config['input']['filename']))

				# Check if theres is some valid QRCode with student info
				found = False
				for s in qrcode_students:
					questions, unique_id_calculated = self.questionsDB.getQuestions(self.selected, uniqueid=None, **s)
					if (unique_id == unique_id_calculated):
						student = s
						found = True
						break
				if not found:
					if self.verbose > 1:
						print("Warning: There is no QRCode with student information. Serching for student in the entire {} list.".format(self.config['input']['filename']))
	
					# Try to find student in the entire student list
					found = False
					for s in self.students:
						questions, unique_id_calculated = self.questionsDB.getQuestions(self.selected, uniqueid=None, **s)
						if (unique_id == unique_id_calculated):
							student = s
							found = True
							break

					if not found:
						# Impossible to find the student info
						if self.verbose > 0:
							print("Eror: Impossible to identify the current student. Just correcting.")
						continue # Go to next answer area
			
			if self.verbose > 2:
				print("Correcting test of {}, question {}.".format(str(student)[13:-2], question_num+1))

			# Get specifc question Class
			question = questions[question_num]

			# Get score for the current question
			score, ansAreaPosProc, ansFeedback = question.doCorrection(ansArea)
			if score is None:
				continue # Impossible to correct the current answer area, go to next answer area

			if self.verbose > 1:
				print("Test of {}, question {}: {}".format(str(student)[13:-2], question_num+1, score))

			# Update the score of current question
			updated_score, all_scores = self.correction.updateScore(student, question_num, score)

			# Show correct answer in image
			if self.verbose > 0:
				import numpy as np
				h,w,_ = ansAreaPosProc.shape
				quad_src = np.float32([[0,0],[w,0],[w,h],[0,h]])
				quad_dst = np.float32([[[x[0],x[1]] for x in contours[0]]])
				ImageUtils.overlayWarpImage(img, ansAreaPosProc, quad_dst, quad_src)

			# Show feedback score in image
			if self.verbose > 0:
				import numpy as np
				topTL, topBL = ImageUtils.findPointsPerpendicularToTheLine(markers["tl"], markers["tr"], markers["tlr"])
				topBR, topTR = ImageUtils.findPointsPerpendicularToTheLine(markers["tr"], markers["tl"], markers["trr"])
				botTL, botBL = ImageUtils.findPointsPerpendicularToTheLine(markers["bl"], markers["br"], markers["blr"])
				botBR, botTR = ImageUtils.findPointsPerpendicularToTheLine(markers["br"], markers["bl"], markers["brr"])
				topTL = ImageUtils.findPointAlongTheLine(topTL, topTR, markers["tlr"]*2); topTR = ImageUtils.findPointAlongTheLine(topTR, topTL, markers["trr"]*2)
				topBL = ImageUtils.findPointAlongTheLine(topBL, topBR, markers["tlr"]*2); topBR = ImageUtils.findPointAlongTheLine(topBR, topBL, markers["trr"]*2)
				botTL = ImageUtils.findPointAlongTheLine(botTL, botTR, markers["blr"]*2); botTR = ImageUtils.findPointAlongTheLine(botTR, botTL, markers["brr"]*2)
				botBL = ImageUtils.findPointAlongTheLine(botBL, botBR, markers["blr"]*2); botBR = ImageUtils.findPointAlongTheLine(botBR, botBL, markers["brr"]*2)
				topQuad = np.float32([[topTL, topTR, topBR, topBL]]); botQuad = np.float32([[botTL, botTR, botBR, botBL]])
				th, tw = (ImageUtils.distance(topTL, topBL)+ImageUtils.distance(topTR, topBR))*0.5, (ImageUtils.distance(topTL, topTR)+ImageUtils.distance(topBL, topBR))*0.5
				bh, bw = (ImageUtils.distance(botTL, botBL)+ImageUtils.distance(botTR, botBR))*0.5, (ImageUtils.distance(botTL, botTR)+ImageUtils.distance(botBL, botBR))*0.5
				th, tw, bh, bw = int(th), int(tw), int(bh), int(bw); topBar = np.zeros((th,tw,3), np.uint8); botBar = np.zeros((bh,bw,3), np.uint8)
				topBar[:] = (200,200,255); botBar[:] = (200,200,255)

				txtStudent = " ".join([v for k,v in student.items()])
				b = th//10; ImageUtils.drawTextInsideTheBox(topBar[b:th-b,b:tw-b,:], txtStudent, color=(255,0,0), thickness=1)
				scores = all_scores.copy()
				for k,v in scores.items(): scores[k] = "?" if v is None else v
				txtScores = ", ".join([str(v) for k,v in scores.items() if k in self.correction.fieldsname_quest])
				txtScores = "{}: {} ({}: {})".format(question_num+1, score, txtScores, scores[self.correction.fieldnames_final[0]])
				b = bh//10; ImageUtils.drawTextInsideTheBox(botBar[b:bh-b,b:bw-b,:], txtScores, color=(255,0,0))

				ImageUtils.overlayWarpImage(img, topBar, topQuad, np.float32([[0,th],[tw,th],[tw,0],[0,0]]))
				ImageUtils.overlayWarpImage(img, botBar, botQuad, np.float32([[0,bh],[bw,bh],[bw,0],[0,0]]))

			# Show feedback on current image: blue if nothing to update and green if some score was updated
			if len(updated_score)>0:
				cv2.drawContours(img, contours,0,(0,255,0), thickness=4)
			else:
				cv2.drawContours(img, contours,0,(255,0,0), thickness=4)

			# Save all updated information in correction folder (image of page, answer area, score, etc)
			for k,v in updated_score.items():
				if self.verbose > 0:
					print("Student ", end="")
					for s,t in student.items():
						print("{}: {} ".format(s,t), end="")
					print("updated {} to {}".format(k, v))

				# Make path to corrction and student
				full_path, full_path_student = self.makeDirectoryForAStrudent(student)
				
				# Save updated information less final score
				if k != self.config['correction']['headers']['final']:
					import os

					# Remove all previous questions' feedback
					from pathlib import Path
					for p in Path(full_path_student).glob(k+"*.jpg"):
						p.unlink()

					# Save image and feedback
					cv2.imwrite(full_path_student+"/"+k+"_"+str(v)+".jpg", ansAreaPosProc)
					if ansFeedback is not None:
						cv2.imwrite(full_path_student+"/"+k+"_Feedback.jpg", ansFeedback)
				
				# Save student score
				import csv
				with open(full_path_student+"/"+self.config['correction']['csv_file'], 'w') as f:
					writer = csv.DictWriter(f, fieldnames=self.correction.fieldnames, **self.correction.csv_param)
					writer.writeheader()
					writer.writerow(all_scores)
					f.close()

		# Save CSV with scores
		self.correction.save()
		return img

	def camera(self, webcam_id):
		if self.verbose > 2:
			print(cv2.getBuildInformation())
		elif self.verbose > 1:
			print("OpenCV version: " + cv2.__version__)

		# Get a webcam
		cap = cv2.VideoCapture(webcam_id)

		# For each frame, try to make correction
		cv2.namedWindow("Press 'space' to close...", cv2.WINDOW_NORMAL)
		while(cap.isOpened()):
			_, i = cap.read()
			i = self.doCorrection(i)
			cv2.imshow("Press 'space' to close...", i)
			if cv2.waitKey(1) & 0xFF == ord(' '):
				break

		# Close camera and windows
		cap.release()
		cv2.destroyAllWindows()

	def readPDF(self, pdf_filename):
		if self.verbose > 2:
			cv2.namedWindow("Press 'space' to next image...", cv2.WINDOW_NORMAL)

		# For each image in PDF input, try to make correction
		for i in Utils.getImagesFromPDF(pdf_filename):
			i = self.doCorrection(i)
			if self.verbose > 2:
				cv2.imshow("Press 'space' to next image...", ImageUtils.conditionalResize(i))
				if cv2.waitKey(0) & 0xFF == ord(' '):
					continue
# MAIN #
########



def main():
	try:
		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument("-v", "--verbose", default=1, action="count", help="Increase output verbosity (most verbose: -vv).")
		parser.add_argument("-s", "--silent", action="store_true", help="Enable silent mode (disable almost all output).")
		parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (show many windows and information).")
		parser.add_argument("-q", "--question_file", type=str, nargs=1, help="Run a specific question file for debug.")
		parser.add_argument("-i", "--interactive", action="store_true", help="Use interative console for debug.")
		parser.add_argument("-t", "--temporary_dir", type=str, nargs=1, help="Directory used by temporary dirs/files. If not used, a temporary directory will be created and deleted in sequence.")
		parser.add_argument("config_file", default="config.json", type=str, nargs='?', help="Configure file input (JSON format).")
		parser.add_argument("-w", "--webcam", choices=range(10), type=int, nargs=1, help="Use webcam with ID.")
		parser.add_argument("-p", "--pdf", type=str, nargs=1, help="PDF file with all scanned tests to correct it.")
		parser.add_argument("-e", "--examples", type=str, choices=[k for k,_ in sorted(examples.items(), key=lambda t: t[0])], help="Echo an example.")
		args = parser.parse_args()

		if args.examples is not None:
			for l in examples[args.examples].split('\n'):
				print(l)
			return

		if args.silent:
			args.verbose = 0

		if args.debug:
			if not __debug__:
				return
			import pdb
			args.verbose = 3
			print(r"""
DEBUG MODE ENABLED
==================
Breakpoint b     Set a breakpoint
Next       n     Execute the next line
Print      p     Print the value of the variable following p
Repeat     Enter Repeat the last entered command
List       l     Show few lines above and below the current line
Step       s     Step into a subroutine
Return     r     Run until the current subroutine returns
Continue   c     Stop debugging the current breakpoint and continue normally
StackUp    u     Navigate up a stack frame
StackDown  d     Navigate down a stack frame
Quit       q     Quit pdb abruptly
""")
			pdb.set_trace()
		else:
			if __debug__: # Execute again, but optimized!
				import os, sys
				os.execl(sys.executable, sys.executable, '-OO', *sys.argv)
				return

		if args.question_file:
			import code
			console = code.InteractiveConsole(locals=locals())
			if args.verbose and args.interactive:
				print("Running '{}' in Interactive Console. Press Ctrl-D to exit.".format(args.question_file[0]))
			console.runcode(open(args.question_file[0]).read())
			if args.interactive:
				console.interact(banner="")
			return

		m = Main(config_file=args.config_file,config_default=examples['config'],verbose=args.verbose,temp_dir= None if args.temporary_dir is None else args.temporary_dir[0])

		if args.webcam:
			m.camera(args.webcam[0])
		elif args.pdf:
			m.readPDF(args.pdf[0])
		else:
			m.makeTests()

	except Exception as e:
		if args.verbose > 2:
			import traceback
			traceback.print_exc()
		elif args.verbose > 1:
			import traceback
			print("(Last 6 traceback lines)")
			for er in traceback.format_exc().splitlines()[-7:]:
				print(er)
		elif args.verbose:
			for er in e.args:
				print (er)
		else:
			print("Failure. Disable silent mode for more information.")



examples = {'config': r'''
//		It is a comment (use '//' at frist postion of line)
//		OBS1: "key": ["Overwrite list"],
//		OBS2: "key+": ["Append"],
//		OBS3: "+key": ["Insert at begin"],
//		OBS4: "key": """First Line<breakline>Second Line""" is equal "key": ["First Line", "Second Line"]

{
	"includeJSON": [
//		"Other json file to attach."
	],
	"questions": {
		"salt": "",
		"db_path": "../Questions",
		"select" : [
//			{"path": "Group",  "weight": 1, "replaces": {"%PREFIX%": "Weight 1"}}
		]
	},
	"input": {
		"filename" : "Students.csv",
		"delimiter": ";",
		"quotechar": "\""
	},
	"output": {
		"tests":    "Tests.pdf",
		"template": "Template.pdf"	
	},
	"correction": {
		"path": "Correction",
		"csv_file": "_scores.csv",
		"delimiter": ";",
		"quotechar": "\"",
		"headers": {
			"student_directory_id": "%NAME%",
			"identification": {"%ID%": "ID", "%NAME%": "Name",  "%EMAIL%": "EMail"},
			"counter": "%COUNT%",
			"intermediate": "Question %COUNT%",
			"final": "Final Score"
		},
		"final_calc": [
			"def final_calc(questions, weight):",
			"  tab = {'F-':0, 'F':26, 'F+':37, 'D-':46, 'D':53, 'D+':59, 'C-':65, 'C':70, 'C+':75, 'B-':80, 'B':84, 'B+':88, 'A-':92, 'A':96, 'A+':100}",
			"  pts = sum([weight[i] * tab[questions[i]] for i in range(len(questions))])//sum(weight)",
			"  for k,v in sorted(tab.items(), key=lambda item: item[1], reverse=True):",
			"    if pts >= v: return k"
		]
	},
	"tex": {
		"max_pages"                                                     : 4,
		"qrcode_id_must_be_concatenated_with_dash_plus_the_page_number" : "%QRCODE_ID%",
		"question_image_answer_area"                                    : "%IMAGE_ANSWER_AREA%",
		"question_counter"                                              : "%COUNT%",
		"question_total"                                                : "%TOTAL%",
		"answer_text"                                                   : "%ANSWER_TEXT%",
		"replaces":{
			"%UNIVERSITY%": "University Name",
			"%TEST_NAME%":  "Test Name",
			"%COURSE%":     "Course Name",
			"%PROFESSOR%":  "Professor Name",
			"%CLASS%":      "Class Name",
			"%DATE%":       "Date",
			"%LOGO_IMG%":   "img/logo.jpeg",
			"%NAME_LABEL%": "Name",
			"%ID_LABEL%":   "ID",
			"%EMAIL_LABEL%": "Email",
			"%PROFESSOR_ABBREVIATION_LABEL%": "Prof.",
			"%SIGNING_ABBREVIATION_LABEL%": "Sign",
			"%INITIALLING_ABBREVIATION_LABEL%": "Initials",
			"%ANSWER_AREA_BEFORE%": "\\textbf{Answer of question %COUNT%:}",
			"%ANSWER_AREA_AFTER%": "\\textbf{\\underline{Attention:}} Completely fill in the corresponding circles and do not shave!",
			"%TEMPLATE_LABEL%": "Template"
		},
		"includes": [
//			"image directory path"
		],
		"preamble": [
			"\\documentclass[twoside,a4paper,11pt]{article}",
			"",
			"\\usepackage[english,brazilian]{babel}",
			"\\usepackage[utf8]{inputenc}",
			"\\usepackage[T1]{fontenc}",
			"\\usepackage{fancyhdr}",
			"\\usepackage{needspace}",
			"\\usepackage{framed}",
			"\\usepackage{color}",
			"\\usepackage{xcolor}",
			"\\usepackage{array}",
			"\\usepackage{tabularx}",
			"\\usepackage{longtable}",
			"\\usepackage{multirow}",
			"\\usepackage{multicol}",
			"\\usepackage{amsmath}",
			"\\usepackage{amsfonts}",
			"\\usepackage{graphicx}",
			"\\usepackage{enumitem}",
			"\\usepackage{tabulary}",
			"\\usepackage{listings}",
			"",
				"% http://texdoc.net/texmf-dist/doc/latex/geometry/geometry.pdf",
				"\\usepackage[includeheadfoot, top=20mm, bottom=10mm, left=15mm, right=15mm, headheight=36mm, headsep=3mm]{geometry}",
				"\\pagestyle{fancy}",
				"\\renewcommand{\\headrulewidth}{0pt} % Remove header line",
				"\\renewcommand{\\footrulewidth}{0pt} % Remove footer line",
				"\\pagenumbering{arabic}",
			"",
			"\\newcommand{\\includeqrcodeimage}[2]{\\includegraphics[width=2.4cm]{#1-#2.png}}",
			"",
			"\\newcommand{\\myemph}[1]{\\textbf{#1}}",
			"\\renewcommand{\\emph}[1]{\\myemph{#1}}",
			"",
			"\\begin{document}",
			""
		],
		"termination": [
			"\\end{document}"
		],
		"test":{
			"header": [
				"\\setcounter{page}{1}",
				"",
				"\\chead{",
					"\\ifnum\\value{page}=1",
						"\\begin{tabular}{|p{2cm}|p{11cm}|p{2.5cm}|}",
						"\\hline",
						"\\multirow{4}{*}{\\includegraphics[width=2cm]{%LOGO_IMG%}} &",
							"\\multicolumn{1}{c|}{{\\LARGE\\textbf{%UNIVERSITY%}}} &",
							"\\multirow{6}{*}{\\includeqrcodeimage{%QRCODE_ID%}{\\thepage}} \\\\",
						" & \\multicolumn{1}{c|}{\\Large\\textbf{%TEST_NAME% \\hspace{1cm} %DATE%}} & \\\\",
						" & \\multicolumn{1}{c|}{\\large\\textbf{%COURSE% \\hspace{1cm} %CLASS%}} & \\\\",
						" & \\multicolumn{1}{c|}{\\normalsize\\textbf{%PROFESSOR_ABBREVIATION_LABEL%} %PROFESSOR%} & \\\\",
						"\\cline{1-2}",
						" \\multicolumn{2}{|c|}{",
							"\\begin{tabular}{p{6cm} p{4cm} p{3cm}}",
							"\\multicolumn{2}{p{10cm}}{\\textbf{%NAME_LABEL%:} %NAME%} & \\textbf{%ID_LABEL%:} %ID% \\\\",
							"\\textbf{%SIGNING_ABBREVIATION_LABEL%} \\underline{\\hspace{5cm}} & \\multicolumn{2}{l}{\\textbf{%EMAIL_LABEL%:} %EMAIL%} \\\\",
							"\\end{tabular}",
						"} & \\\\",
						"\\hline",
						"\\end{tabular}",
					"\\else",
						"\\begin{tabular}{|p{2cm}|p{11cm}|p{2.5cm}|}",
						"\\hline",
						"\\multirow{3}{*}{\\includegraphics[width=2cm]{%LOGO_IMG%}} &",
							"\\multicolumn{1}{c|}{{\\LARGE\\textbf{%UNIVERSITY%}}} &",
							"\\multirow{3}{*}{\\includeqrcodeimage{%QRCODE_ID%}{\\thepage}} \\\\",
						" & \\multicolumn{1}{c|}{\\Large\\textbf{%TEST_NAME% \\hspace{1cm} %DATE%}} & \\\\",
						" & \\multicolumn{1}{c|}{\\large\\textbf{%COURSE% \\hspace{1cm} %CLASS%}} & \\\\",
						" & \\multicolumn{1}{c|}{",
							"\\begin{tabular}{p{4cm} p{4cm} p{3cm}}",
							"\\multicolumn{2}{p{8cm}}{\\textbf{%NAME_LABEL%:} %NAME%} & \\textbf{%ID_LABEL%:} %ID% \\\\",
							"\\textbf{%INITIALLING_ABBREVIATION_LABEL%:} \\underline{\\hspace{2cm}} & \\multicolumn{2}{l}{\\textbf{%EMAIL_LABEL%:} %EMAIL%} \\\\",
							"\\end{tabular}",
						"} & \\\\",
						"\\hline",
						"\\end{tabular}",
					"\\fi",
				"}",
				"",
				"\\cfoot{",
					"\\thepage",
				"}",
				"",
				"",
				"\\vspace{0cm}",
				""
			],
			"before": [
				"",
				"\\needspace{1\\baselineskip}",
				"\\textbf{Questão %COUNT% de %TOTAL% (%PREFIX%):}"
			],
			"after": [
				"",
				"\\noindent\\makebox[\\textwidth]{\\begin{tabular}{c}",
				"%ANSWER_AREA_BEFORE% \\\\",
				"\\includegraphics[width=.9\\textwidth]{%IMAGE_ANSWER_AREA%} \\\\",
				"%ANSWER_AREA_AFTER% \\\\",
				"\\end{tabular}}"
			],
			"footer": [
				"\\cleardoublepage{}"
			]
		},
		"template":{
			"header": [
				"",
				"\\begin{center}",
				"\\begin{tabular}{|c c|}",
				"\\hline",
				"\\multicolumn{2}{|c|}{ {\\LARGE\\textbf{%TEMPLATE_LABEL%}} } \\\\",
				"{\\large\\textbf{%COURSE%}} & {\\large\\textbf{%TEST_NAME%}} \\\\",
				"{\\large\\textbf{%CLASS%}} & {\\large\\textbf{%DATE%}} \\\\",
				"\\hline",
				"\\end{tabular}",
				"\\end{center}",
				""
			],
			"before": [
				"\\begin{tabulary}{\\textwidth}{|p{0cm}*{%TOTAL%}{|L}|}",
				"\\hline",
				"\\multirow{2}{*}{} & \\multicolumn{%TOTAL%}{|c|}{\\textbf{%NAME% (%ID%)}} \\\\",
				"\\hline"
			],
			"answer": [
				" & {\\scriptsize\\textbf{%COUNT%:}} {\\small %ANSWER_TEXT%}"
			],
			"after": [
				" \\\\ ",
				"\\hline",
				"\\end{tabulary}",
				""
			],
			"footer": [
				""
			]
		}
	}
}
'''
, 'assay': r"""
from MakeTests import QuestionEssay
class QualquerNome(QuestionEssay):
	def makeVariables(self):
		self.setLabels(scores=["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+", "F", "F-"], blank_warning="--- Deixar em branco! ---")
	def getQuestionTex(self):
		tex = '''Esta é uma questão dissertativa.
'''
		lines_answer_area = 10
		tex += "\n\n\\begin{tabularx}{\\textwidth}{|X|}\\hline"
		for l in range(lines_answer_area): tex += (" \\\\")
		tex += "\\hline\\end{tabularx}"
		return tex
	def getAnswerText(self,LaTeX):
		return ""
"""
, 'choices': r"""
from MakeTests import QuestionMultipleChoice
class QualquerNome(QuestionMultipleChoice):
	def makeSetup(self):
		import random
		questFacil = [
			{  "statement": "Enunciado Facil 1.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()), "itensPerRow":2 },
			{  "statement": "Enunciado Facil 2.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()), "itensPerRow":2 },
			{  "statement": "Enunciado Facil 3.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()), "itensPerRow":2 },
		]; random.shuffle(questFacil)
		questDificil = [
			{  "statement": "Enunciado Dificil 1.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()) },
			{  "statement": "Enunciado Dificil 2.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()) },
			{  "statement": "Enunciado Dificil 3.", "alternatives": sorted([
				["Correta", True],
				["Errada", False],
				["Errada", False],
				["Errada", False],
			], key=lambda k: random.random()) },
		]; random.shuffle(questDificil)
		self.questionDescription  = "Escolha a alternativa correta das questões abaixo:"
		self.questions  = questFacil[0:2] + questFacil[0:2]
		self.scoreTable = [(20,"A+"), (19,"A"), (18,"A-"), (17,"B+"), (16,"B"), (15,"B-"), (14,"C"), (13,"C-"), (12,"D+"), (11,"D"), (0, "F")]
		self.labels     = {"score": "Conceito", "if": "se", "else": "senão", "number_of_right_questions": "Qtd. de questões certas"}
		self.conditionByLineLabel = 3
"""
, 'truefalse': r"""
from MakeTests import QuestionTrueOrFalse
class QualquerNome(QuestionTrueOrFalse):
	def makeSetup(self):
		import random
		grp1 = [
			["Esta mensagem 1 do grupo 2 é Verdadeira", True],
			["Esta mensagem 2 do grupo 2 é Falsa", False],
			["Esta mensagem 3 do grupo 2 é Verdadeira", True],
			["Esta mensagem 4 do grupo 2 é Falsa", False],
			["Esta mensagem 5 do grupo 2 é Verdadeira", True],
			["Esta mensagem 6 do grupo 2 é Falsa", False]
		]; random.shuffle(grp1)
		grp2 = [
			["Esta mensagem 1 do grupo 2 é Verdadeira", True],
			["Esta mensagem 2 do grupo 2 é Falsa", False],
			["Esta mensagem 3 do grupo 2 é Verdadeira", True],
			["Esta mensagem 4 do grupo 2 é Falsa", False],
			["Esta mensagem 5 do grupo 2 é Verdadeira", True],
			["Esta mensagem 6 do grupo 2 é Falsa", False]
		]; random.shuffle(grp2)
		grp3 = [
			["Esta mensagem 1 do grupo 3 é Verdadeira", True],
			["Esta mensagem 2 do grupo 3 é Falsa", False],
			["Esta mensagem 3 do grupo 3 é Verdadeira", True],
			["Esta mensagem 4 do grupo 3 é Falsa", False],
			["Esta mensagem 5 do grupo 3 é Verdadeira", True],
			["Esta mensagem 6 do grupo 3 é Falsa", False]
		]; random.shuffle(grp3)

		all = grp1[:3] + grp2[:3] + grp3[:3]
		random.shuffle(all)

		self.questions            = all
		self.questionDescription  = "Referente aos Capítulos 3, 4 e 5, preencha a tabela abaixo com V ou F de acordo com as perguntas a seguir."
		self.scoreTable           = [(20,"A+"), (19,"A"), (18,"A-"), (17,"B+"), (16,"B"), (15,"B-"), (14,"C"), (13,"C-"), (12,"D+"), (11,"D"), (0, "F")]
		self.wrongFactor          = 3
		self.labels               = {"true": "V", "false": "F", "score": "Conceito", "if": "se", "else": "senão", "number_of_right_questions": "Questões certas", "number_of_wrong_questions": "Questões erradas"}
		self.conditionByLineLabel = 3
"""
, 'questionanswer': r"""
from MakeTests import QuestionQA
class QualquerNome(QuestionQA):
	def makeSetup(self):
		self.full_list_question_answer_text = [
			[''' Pergunta 01 ''', '''Resposta 01 '''],
			[''' Pergunta 02 ''', '''Resposta 02 '''],
			[''' Pergunta 03 ''', '''Resposta 03 '''],
			[''' Pergunta 04 ''', '''Resposta 04 '''],
			[''' Pergunta 05 ''', '''Resposta 05 '''],
			[''' Pergunta 06 ''', '''Resposta 06 '''],
			[''' Pergunta 07 ''', '''Resposta 07 '''],
			[''' Pergunta 08 ''', '''Resposta 08 '''],
			[''' Pergunta 09 ''', '''Resposta 09 '''],
			[''' Pergunta 10 ''', '''Resposta 10 '''],
			[''' Pergunta 11 ''', '''Resposta 11 '''],
			[''' Pergunta 12 ''', '''Resposta 12 '''],
		]
		self.list_question_answer_index = QuestionQA.auxShuffle(self.full_list_question_answer_text, 8)
		self.question_description = "Referente ao Capítulo 1, associe as Perguntas com as Respostas."
		self.scoreTable = [(8,"A"), (7,"B"), (6,"B-"), (5,"C"), (4,"D"), (0,"F")]
		self.labels     = {"questions": "Perguntas", "answers": "Respostas", "score": "Conceito", "if": "se", "else": "senão", "number_of_right_questions": "Qtd. de questões certas"}
		self.conditionByLineLabel = 3
"""
, 'number': r"""
from MakeTests import QuestionNumber
class QualquerNome(QuestionNumber):
	def makeSetup(self):
		self.max_digits = 6
		self.decimal_separator = ","
		self.hlabel = "- - - Digitos (selecionar apenas um por linha) - - -"
		self.vlabel = "> Seq. digitos >"
		self.expected_value = 123.4
	def getQuestionTex(self):
		return "Escreva {}".format(self.expected_value)
	def getAnswerText(self,LaTeX):
		return str(self.expected_value)
	def getScoreFromNumber(self, num):
		if num is None:
			return "F"
		else:
			return "A" if num == self.expected_value else "D"
"""
, 'ocr': r"""
from MakeTests import QuestionOCR
class QualquerNome(QuestionOCR):
	def makeVariables(self):
		pass
	def getQuestionTex(self):
		return "Escreva 123"
	def answerAreaAspectRate(self):
		return 4/1
	def getAnswerText(self,LaTeX):
		return "123"
	def getScore(self, text):
		return "A" if text == "123" else "F"
"""
}

if __name__ == "__main__":
	main()
