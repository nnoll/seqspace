site:
	rsync -rk ../build/ .
	for path in autoencode inference normalize; do \
		sed -i 's|assets/drosophila|seqspace/assets/drosophila|g' sci/$$path/index.html; \
		sed -i 's|assets/autoencode|seqspace/assets/autoencode|g' sci/$$path/index.html; \
		sed -i 's|assets/swissroll|seqspace/assets/swissroll|g'   sci/$$path/index.html; \
	done
