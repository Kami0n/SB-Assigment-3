def get_annotations(annot_f):
	d = {}
	with open(annot_f) as f:
		lines = f.readlines()
		for line in lines:
			(key, val) = line.split(',')
			# keynum = int(self.clean_file_name(key))
			d[key] = int(val)
	return d

def clean_file_name(fname):
	return fname.split('/')[1].split(' ')[0]