MAKEFLAGS += --no-builtin-rules

BENCH_ROWS := 1000000
BENCH_COLS := 25

BENCH_INPUTS := mixed/input.tsv all_numeric/input.tsv all_character/input.tsv
BENCH_SRC := $(wildcard */*.R)
BENCH_MARKS := $(filter-out $(BENCH_INPUTS:.tsv=.R),$(BENCH_SRC))
BENCH_MARKS_TSV := $(BENCH_MARKS:.R=.tsv)
BENCH_OUT := mixed.tsv all_numeric.tsv all_character.tsv

all: $(BENCH_OUT) session_info.tsv

$(BENCH_OUT): $(BENCH_MARKS_TSV)
	Rscript summarise-benchmarks.R

session_info.tsv: session_info.R
	./$<

lib/vroom: setup.R
	Rscript $<
	touch $@

%/input.tsv: %/input.R
	Rscript $< $(BENCH_ROWS) $(BENCH_COLS) $@

# vroom_cran benchmarks depend on lib/vroom being installed
vroom_cran-%.tsv: vroom_cran-%.R lib/vroom
	./run-bench.R $< $@ $(@D)/input.tsv

%.tsv: %.R
	./run-bench.R $< $@ $(@D)/input.tsv

clean:
	rm -f $(BENCH_INPUTS) $(BENCH_MARKS_TSV) $(BENCH_OUT) session_info.tsv
	rm -rf lib/
