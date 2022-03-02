site:
	rsync -rk ../build/ .
	sed -i 's|assets/|seqspace/assets/|g' sci/autoencode/index.html
	sed -i 's|assets/|seqspace/assets/|g' sci/inference/index.html
	sed -i 's|assets/|seqspace/assets/|g' sci/normalize/index.html
