# housekeeping
.PHONY: all archive clean documentation
.SUFFIXES:
.SECONDARY:

# variables
DATA=$(HOME)/root/data/seqspace
MODELS=
NORMED=

DIRECTORIES=\
	drosophila

# commands
JULIA     = julia --project=.
NORMALIZE = @echo ">generating $(@:$(DATA)/%=%)";\
			$(JULIA) bin/normalize.jl
FITMODEL  = @echo ">generating $(@:$(DATA)/%=%)";\
			$(JULIA) bin/fitmodel.jl

# recipe template
define RULE
$(DATA)/$(DIR)/model/norms.jld2: param/$(DIR)/normalize.jl $(shell find $(DATA)/$(DIR)/raw -type f)
	$$(NORMALIZE) -o $$@ -p $$< $(DATA)/$(DIR)/raw
$(DATA)/$(DIR)/model/model.jld2: param/$(DIR)/model.jl $(DATA)/$(DIR)/model/norms.jld2
	$$(FITMODEL) -o $$@ -p $$^

MODELS+=$(DATA)/$(DIR)/model/model.jld2
NORMED+=$(DATA)/$(DIR)/model/norms.jld2


endef

all: models

# generate individual data rules
data.mk: Makefile
	$(file > $@,) $(foreach DIR,$(DIRECTORIES),$(file >> $@,$(RULE)) )

-include data.mk

normed: $(NORMED)
models: $(MODELS)

archive:
	@echo ">archiving current models";\
	$(JULIA) bin/archive.jl $(DATA) $(DIRECTORIES)

documentation:
	cd docs && $(JULIA). make.jl

# TODO: clean up figures
clean:
	@echo ">removing models";\
	rm -f $(NORMED) $(MODELS)
