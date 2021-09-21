# housekeeping
.PHONY: all
.SUFFIXES:

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
$(DATA)/$(DIR)/model/model.bson: $(DATA)/$(DIR)/model/norms.jld2
	$$(FITMODEL)

MODELS+=$(DATA)/$(DIR)/model/model.bson
NORMED+=$(DATA)/$(DIR)/model/norms.jld2


endef

all: normed

# generate individual data rules
data.mk: Makefile
	$(file > $@,) $(foreach DIR,$(DIRECTORIES),$(file >> $@,$(RULE)) )

-include data.mk

normed: $(NORMED)
models: $(MODELS)
