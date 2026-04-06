# ============================================================
# Makefile — Finance kernel cache measurement suite
#
# Builds all PAPI-instrumented kernels:
#   1. mm_papi     — naive dense GEMM (12 variants: 2 layouts × 6 orders)
#   2. cholesky    — Cholesky factorisation (4 variants: 2 layouts × 2 algos)
#   3. mc_paths    — correlated MC path generation (2 variants: 2 layouts)
#   4. garch_mle   — GARCH(1,1) log-likelihood (1 variant, no layout)
#
# On BigRed200:
#   module load papi
#   make all
# ============================================================

CC      ?= gcc
CFLAGS  := -O3 -march=native -Wall -Wextra
LDFLAGS := -lm

# Use pkg-config for PAPI paths (works on BigRed200)
PAPI_CFLAGS := $(shell pkg-config --cflags papi 2>/dev/null)
PAPI_LIBS   := $(shell pkg-config --libs papi 2>/dev/null || echo "-lpapi")
CFLAGS  += $(PAPI_CFLAGS)
LDFLAGS := $(PAPI_LIBS) $(LDFLAGS)

# Directory for binaries
BINDIR  := bin
$(shell mkdir -p $(BINDIR))

# ---- mm_papi (existing kernel) ----------------------------------
MM_LAYOUTS := ROW_MAJOR COL_MAJOR
MM_ORDERS  := ORDER_IJK ORDER_IKJ ORDER_JIK ORDER_JKI ORDER_KIJ ORDER_KJI

MM_BINS :=
define mm_rule
  MM_BINS += $(BINDIR)/mm_$(1)_$(2)
  $(BINDIR)/mm_$(1)_$(2): mm_papi.c
	$$(CC) $$(CFLAGS) -D$(1) -D$(2) -o $$@ $$< $$(LDFLAGS)
endef

$(foreach lay,$(MM_LAYOUTS),\
  $(foreach ord,$(MM_ORDERS),\
    $(eval $(call mm_rule,$(lay),$(ord)))))

# ---- cholesky_papi -----------------------------------------------
CHOL_LAYOUTS := ROW_MAJOR COL_MAJOR
CHOL_ALGOS   := ALGO_BANACHIEWICZ ALGO_CROUT

CHOL_BINS :=
define chol_rule
  CHOL_BINS += $(BINDIR)/cholesky_$(1)_$(2)
  $(BINDIR)/cholesky_$(1)_$(2): cholesky_papi.c
	$$(CC) $$(CFLAGS) -D$(1) -D$(2) -o $$@ $$< $$(LDFLAGS)
endef

$(foreach lay,$(CHOL_LAYOUTS),\
  $(foreach algo,$(CHOL_ALGOS),\
    $(eval $(call chol_rule,$(lay),$(algo)))))

# ---- mc_paths_papi -----------------------------------------------
MC_BINS := $(BINDIR)/mc_paths_ROW_MAJOR $(BINDIR)/mc_paths_COL_MAJOR

$(BINDIR)/mc_paths_ROW_MAJOR: mc_paths_papi.c
	$(CC) $(CFLAGS) -DROW_MAJOR -o $@ $< $(LDFLAGS)

$(BINDIR)/mc_paths_COL_MAJOR: mc_paths_papi.c
	$(CC) $(CFLAGS) -DCOL_MAJOR -o $@ $< $(LDFLAGS)

# ---- garch_mle_papi (no layout dimension) ------------------------
GARCH_BINS := $(BINDIR)/garch_mle

$(BINDIR)/garch_mle: garch_mle_papi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ---- aggregate targets -------------------------------------------
ALL_BINS := $(MM_BINS) $(CHOL_BINS) $(MC_BINS) $(GARCH_BINS)

.PHONY: all mm cholesky mc_paths garch clean

all:       $(ALL_BINS)
mm:        $(MM_BINS)
cholesky:  $(CHOL_BINS)
mc_paths:  $(MC_BINS)
garch:     $(GARCH_BINS)
finance:   $(CHOL_BINS) $(MC_BINS) $(GARCH_BINS)

clean:
	rm -rf $(BINDIR)

# Print all targets (for scripting)
.PHONY: list
list:
	@echo $(ALL_BINS)
