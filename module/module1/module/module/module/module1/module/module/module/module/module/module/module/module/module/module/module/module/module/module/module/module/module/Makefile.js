.PHONY: deps_table_update modified_only_fixup extra_style_checks quality style fixup fix-copies test test-examples

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := examples scripts src tests utils

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black --preview $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Update src/diffusers/dependency_versions_table.py

deps_table_update:
	@python setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/diffusers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# autogenerating code

autogenerate_code: deps_table_update

# Check that the repo is in a good state

repo-consistency:
	python utils/check_dummies.py
	python utils/check_repo.py
	python utils/check_inits.py

# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)
	doc-builder style src/diffusers docs/source --max_len 119 --check_only --path_to_docs docs/source

# Format source code automatically and check is there are any problems left that need manual fixing

extra_style_checks:
	python utils/custom_init_isort.py
	doc-builder style src/diffusers docs/source --max_len 119 --path_to_docs docs/source

# this target runs checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
	${MAKE} autogenerate_code
	${MAKE} extra_style_checks

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup extra_style_checks autogenerate_code repo-consistency

# Make marked copies of snippets of codes conform to the original

fix-copies:
	python utils/check_dummies.py --fix_and_overwrite

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run tests for examples

test-examples:
	python -m pytest -n auto --dist=loadfile -s -v ./examples/pytorch/


# Release stuff

pre-release:
	python utils/release.py

pre-patch:
	python utils/release.py --patch

post-release:
	python utils/release.py --post_release

post-patch:
	python utils/release.py --post_release --patch

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

OS := $(shell uname -s)

ifeq ($(OS), Linux)
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS), Darwin)
  NPROCS := 2
else
  NPROCS := 0
endif # $(OS)

ifeq ($(NPROCS), 2)
	CONCURRENCY := 2
else ifeq ($(NPROCS), 1)
	CONCURRENCY := 1
else ifeq ($(NPROCS), 3)
	CONCURRENCY := 3
else ifeq ($(NPROCS), 0)
	CONCURRENCY := 0
else
	CONCURRENCY := $(shell echo "$(NPROCS) 2" | awk '{printf "%.0f", $$1 / $$2}')
endif

# You can set this variable from the command line.
SPHINXOPTS    =

.PHONY: lint mypy style black test test_ci spell copyright html doctest clean_sphinx coverage coverage_erase clean

all_check: spell style lint copyright mypy clean_sphinx html doctest

lint:
	pylint -rn qiskit_machine_learning test tools
	python tools/verify_headers.py qiskit_machine_learning test tools
	python tools/find_stray_release_notes.py

mypy:
	mypy qiskit_machine_learning test tools

style:
	black --check qiskit_machine_learning test tools docs

black:
	black qiskit_machine_learning test tools docs

test:
	python -m unittest discover -v test

test_ci:
	echo "Detected $(NPROCS) CPUs running with $(CONCURRENCY) workers"
	stestr run --concurrency $(CONCURRENCY)

spell:
	pylint -rn --disable=all --enable=spelling --spelling-dict=en_US --spelling-private-dict-file=.pylintdict qiskit_machine_learning test tools
	sphinx-build -M spelling docs docs/_build -W -T --keep-going $(SPHINXOPTS)

copyright:
	python tools/check_copyright.py

html:
	sphinx-build -M html docs docs/_build -W -T --keep-going $(SPHINXOPTS)

doctest:
	sphinx-build -M doctest docs docs/_build -W -T --keep-going $(SPHINXOPTS)

clean_sphinx:
	make -C docs clean
	
coverage:
	coverage3 run --source qiskit_machine_learning -m unittest discover -s test -q
	coverage3 report

coverage_erase:
	coverage erase

clean: clean_sphinx coverage_erase; 

// Special build system used for Halfix builds. Supports the following features:
//  - All built object files go in build/objs (no cluttering src/)
//  - Build multiple targets at once (i.e. switch between Emscripten and native
//  builds without having to clean)
//  - Parallel build (same as make -j)
//  - Certain files can be conditionally built based on the target
//  - Dependency resolution and conditional compilation (note: this is a bit
//  buggy)
//  - Automatic dependency regeneration (via redep)
//  - Simple to use (no configuration, multi-platform, etc.)
// Usage:
//  node makefile.js [target] [options]
// Omit "target" for default native build.
// Run with --help for options and target list.

var fs = require("fs");
var child_process = require("child_process");
var os = require("os");

function load_json(e) {
    return JSON.parse(fs.readFileSync("build/" + e + "-files.json"));
}

var files = JSON.parse(fs.readFileSync("build/files.json"));
for (var i = 0; i < files.length; i++) {
    var nm = files[i];
    files[i] = load_json(nm);
    files[i].__name = nm;
}

var bits = os.arch() === "x64" ? 64 : 32; // Add your architecture here!
var flags = ["-Wall", "-Wextra", "-Werror", "-g3", "-std=c99"];
var end_flags = [], fincc_flags = [];

// flags.push.apply(flags, "-I/usr/include/SDL -D_GNU_SOURCE=1
// -D_REENTRANT".split(" "));
// flags.push.apply(flags, "-L/usr/lib/x86_64-linux-gnu -lSDL -lSDLmain".split("
// "));
end_flags = "-lSDL -lSDLmain -lm -lz".split(" ");

if (os.endianness() === "BE") {
    console.warn("WARNING: This emulator has not been tested on big-endian platforms and may not work.");
    flags.push("-DCFG_BIG_ENDIAN");
}

// XXX: capture live output from allegro-config --shared
//end_flags.push("-lalleg");

function getFlagsFromExec(a) {
    var child = child_process.execSync(a);
    console.log(child.toString());
    return child.toString().replace(/\n/g, "").split(" ");
}
function merge(dst, src) {
    for (var i = 0; i < src.length; i = i + 1 | 0)dst.push(src[i]);
}

var build32 = false;
var verbose = false;
var cc = "gcc",
    fincc = "gcc";
var build_type = "native";

var optimization = "-O0";

var result = null;

var wasm = 0;

var argv = process.argv.slice(2);
for (var i = 0; i < argv.length; i++) {
    switch (argv[i]) {
        case "redep":
            recompute_dependencies();
            console.log("Successfuly recalculated dependencies");
            process.exit(0);
            break;
        case "clean":
            console.log("Removing all built files in build/");
            child_process.execSync("rm build/objs/*.o");
            process.exit(0);
            break;
        case "gtk":
            merge(flags, getFlagsFromExec("pkg-config --cflags gtk+-3.0"));
            merge(end_flags, getFlagsFromExec("pkg-config --libs gtk+-3.0"));
            build_type = "gtk";
            break;
        case "win32":
            build_type = "win32";
            end_flags.push("-lgdi32", "-lcomdlg32");
            break;
        case "libcpu":
            build_type = "libcpu";
            files[0] = {};
            files[2] = {};
            flags.push("-fPIC", "-shared");
            console.log(build_type);
            flags.push("-DLIBCPU");
            break;
        case "libcpu-wasm":
            build_type = "libcpu";
            result = "libcpu.wasm";
            files[0] = {};
            files[2] = {};
            flags.push("-fPIC");
            cc = fincc = "emcc";
            flags.push("-s", "SIDE_MODULE=1");
            flags.push("-DLIBCPU");
            break;
        case "release":
            argv.push("--optimization-level", "3");
            argv.push("--disable-debug");
            break;
        case "--optimization-level":
            if (isNaN(parseInt(argv[i + 1])))
                optimization = "-O";
            else
                optimization = "-O" + (argv[++i]);
            break;
        case "--verbose":
            verbose = true;
            break;
        case "--32-bit":
            flags.push("-m32");
            break;
        case "--instrument":
            flags.push("-DINSTRUMENT");
            break;
        case "--profile":
            end_flags.push("-pg");
            break;
        case "--output":
            result = argv[++i];
            break;
        case "emscripten":
            result = "halfix.js";
            fincc = cc = "emcc";
            build_type = "emscripten";

            // List appropriate flags
            var my_flags = "";
            my_flags = my_flags.split(" ");
            end_flags.push("-s", "NO_FILESYSTEM=1",
                "-s", "TOTAL_MEMORY=256MB"
                //"-s", "ASSERTIONS=1",
                //"-s", "SAFE_HEAP=1"    
            );
            for (var j = 0; j < my_flags.length; j++) end_flags.push(my_flags[j]);
            break;
        case "--enable-wasm":
            wasm = 1;
            break;
        case "--cc":
            cc = fincc = argv[++i];
            break;
        case "--fincc":
            fincc = argv[++i];
            break;
        case "--disable-debug":
            flags.splice(flags.indexOf("-g3"), 1);
            break;
        case "--help":
            console.log("Actions:");
            console.log(
                "  redep                     Recalculate dependencies (useful for building)");
            console.log("  clean                     Remove all build files");
            console.log("\nTargets (besides native):");
            console.log("  emscripten                Build Emscripten target");
            console.log("\nBuild types:");
            console.log("  release                   Build fastest possible executable");
            console.log("\nOptions:");
            console.log(" --verbose                  Verbose logging during build");
            console.log(" --32-bit                   Build 32-bit target");
            console.log(
                " --optimization-level [n]   Set optimization level of build to n (leave empty for -O)");
            console.log(" --output [path]            Set output file to path");
            console.log(
                " --instrument               Enable instrumentation callbacks");
            console.log(" --profile                  Compile with -pg");
            console.log(" --disable-debug            Compile without debugging information");
            console.log(" --enable-wasm              Compile for WASM target");
            process.exit(0);
        default:
            console.error(
                "Unknown option " + argv[i] +
                ". Run with --help to see all available options.");
            process.exit(1);
    }
}
if (!result) result = "halfix";


if (build_type === "emscripten") {
    end_flags.push("-s", "WASM=" + wasm);
    // Emscripten no longer supports dynCall, apparently. 
    // Should've known better than to rely on an internal, undocumented method. 
    //end_flags.push(
    //    "-s", "\"EXTRA_EXPORTED_RUNTIME_METHODS=['dynCall_vii']\"");
    console.log(end_flags);
}

// https://github.com/emscripten-core/emscripten/issues/5659
if (build_type === "emscripten")
    for (var i = 0; i < flags.length; i++) {
        if (flags[i] === "-std=c99") flags[i] = "-std=gnu99";
        if (end_flags.indexOf("WASM=1") !== -1) {
            //if (flags[i] === "-g3") flags[i] = "-g4";
        }
    }

if (result.indexOf(".js") !== -1 && build_type == "libcpu") {
    for (var i = 0; i < flags.length; i++)
        if (flags[i] === "-shared") flags[i] = "";
    flags.push("-s", "STANDALONE_WASM=1");
}
if (result.indexOf(".js") !== -1 || result.indexOf(".wasm") !== -1) {
    end_flags.splice(end_flags.indexOf("-lSDLmain"), 1);
    end_flags.splice(end_flags.indexOf("-lz"), 1);
}
flags.push("-D" + build_type.toUpperCase() + "_BUILD");

/*
if (optimization !== 0) {
    var i = flags.indexOf("-g3");
    flags[i] = "";
}*/

flags.push(optimization);

var path = require("path");

// A relatively unique, but distinct path.
// Combines the hash of the path, real name, and file flags
function find_object_file_name(fn) {
    var name = path.basename(fn, ".c");
    var dirname = path.dirname(fn);
    var result = 0;
    for (var i = 0; i < dirname.length; i++) {
        result += dirname.charCodeAt(i) | 0;
    }
    return "build/objs/" + result.toString(36) + "-" + name + "-" +
        find_file_append() + ".o";
}

function find_file_append() {
    var id = 0;
    // Based on the current flags, we generate a unique number.
    // To make it as short is possible, we use the base 36 encoding.
    if (flags.indexOf("-m32") !== -1) { // Separate 32-bit build.
        id |= 0x80000000;
    }

    // ADD YOUR FLAGS HERE
    if (flags.indexOf("-g3") !== -1) id |= 1;
    if (flags.indexOf("-O0") !== -1) id |= 2;
    if (flags.indexOf("-O1") !== -1) id |= 4;
    if (flags.indexOf("-O2") !== -1) id |= 8;
    if (flags.indexOf("-O3") !== -1) id |= 16;
    if (cc === "emcc") id |= 32;
    if (flags.indexOf("-pie") !== -1) id |= 64;
    if (flags.indexOf("-DINSTRUMENT") !== -1) id |= 128;
    if (flags.indexOf("-O") !== -1) id |= 256;
    if (flags.indexOf("-DLIBCPU") !== -1) id |= 512;
    if (flags.indexOf("SIDE_MODULE=1") !== -1) id |= 1024;

    // Hash the name of the build
    var x = 0;
    for (var i = 0; i < build_type.length; i++)
        x = (x + build_type.charCodeAt(i)) | 0;
    x ^= build_type.length;
    id |= x << 11;
    return id.toString(36);
}

var to_compile = [];
// Find a list of the files in our directory, so we know what we're looking at.
var build_files = fs.readdirSync("build/");

var all_files = [];

function run_pre_cmd(file_entry) {
    for (var i = 0; i < file_entry.tasks.length; i++) {
        child_process.execSync(file_entry.tasks[i]);
    }
}

function prune_out_similar() {
    var m = -1;
    for (var i = 0; i < flags.length; i++) {
        if (flags[i] === "-m32" || flags[i] === "-m64") {
            if (m !== -1) flags[m] = "";
            m = i;
        }
    }
}

// Create a list of which files we should be looking at. Note that this function
// may be called more than once if our flags change (i.e. we're compiling for a
// 64-bit target and all of a sudden compile.c forces us to go 32-bit)
function check_comp(obj) {
    top: for (var name in obj) {
        if (name == "description" || name == "archive" || name == "__name")
            continue;
        var file_entry = obj[name];
        // First, go through our files and see if we should compile them or not.
        var adf = file_entry.additional_flags;

        var additional_args = [];
        for (var i = 0; i < adf.length; i++) {
            var additional_flags = adf[i];
            if (additional_flags[0] === "@") {
                var toks = additional_flags.split("=");
                switch (toks[0]) {
                    case "@flags":
                        var parts = toks[1].split("|");
                        var noparts = [];
                        for (var j = 0; j < parts.length; j++) {
                            if (parts[j][0] === "!") {
                                noparts.push(parts[j].substring(1));
                                parts[j] = null;
                            }
                        }
                        if (parts.length == 1 && parts[0] === null) parts = [];
                        if (parts.length === 0) parts.push(build_type);
                        var x = false;
                        for (var j = 0; j < parts.length; j++) {
                            x = x || (parts[j]);
                        }
                        if (!x) parts.push(build_type);
                        // console.log(parts, noparts, build_type);
                        if (parts.indexOf(build_type) === -1 ||
                            noparts.indexOf(build_type) !== -1) {
                            if (verbose)
                                console.log(
                                    "Skipping file " + name + " because build type is \"" +
                                    build_type + "\"");
                            continue top;
                        }
                        break;
                    case "@options":
                        prune_out_similar();
                        var parts = toks[1].split("|");
                        var noparts = [];
                        for (var j = 0; j < parts.length; j++) {
                            if (parts[j][0] === "!") {
                                noparts.push(parts[j].substring(1));
                                parts[j] = null;
                            }
                        }
                        for (var j = 0; j < parts.length; j++) {
                            if (flags.indexOf(parts[j]) === -1) {
                                if (verbose)
                                    console.log(
                                        "Skipping file " + name +
                                        " because build does not include flag \"" + parts[j] +
                                        "\"");
                                continue top;
                            }
                        }
                        for (var j = 0; j < parts.length; j++) {
                            console.log
                            if (flags.indexOf(noparts[j]) !== -1) {
                                if (verbose)
                                    console.log(
                                        "Skipping file " + name +
                                        " because build does  include flag \"" + parts[j] + "\"");
                                continue top;
                            }
                        }
                        break;
                    case "@use":
                        // Check if our flags are mutually exclusive
                        if (flags.indexOf(toks[1]) === -1) {
                            if (verbose) {
                                console.log(
                                    "Recomputing files to compile to account for " + toks[1]);
                            }
                            flags.push(toks[1]);
                            // Uh oh... we have a problem.
                            // We have to go back and recompute the files to find.
                            to_compile = [];
                            all_files = [];
                            throw "recompute";
                        }
                        break;
                }
            } else {
                additional_args.push(additional_flags);
            }
        }

        all_files.push(name);
        var objn = find_object_file_name(name);
        // Check if the file exists.
        var d = fs.existsSync(objn);
        if (!d) {
            run_pre_cmd(file_entry);
            if (verbose)
                console.log("Rebuilding because file " + objn + " does not exist");
            to_compile.push({
                name: name,
                additional_args: additional_args,
                includes: file_entry.include_paths
            });
            continue top;
        }

        // Check if we've modified the file since then
        var obj_mtime = get_mtime(objn);

        // Get the mtime
        var fe_mtime = get_mtime(name);
        if (obj_mtime < fe_mtime) {
            run_pre_cmd(file_entry);
            if (verbose)
                console.log(
                    "Rebuilding " + name + " because the object file (" + objn +
                    ") was modified before the file");
            // console.log("obj", new Date(obj_mtime), "dd", new Date(fe_mtime));
            to_compile.push({
                name: name,
                additional_args: additional_args,
                includes: file_entry.include_paths
            });
            continue top;
        }
        // Get its dependencies.
        var deps = file_entry.dependencies;
        for (var i = 0; i < deps.length; i++) {
            var dep_mtime = get_mtime(deps[i]);
            if (dep_mtime > obj_mtime) {
                // UH OH... we have updated our dependency.
                run_pre_cmd(file_entry);
                if (verbose)
                    console.log("Rebuilding " + name + " because dependency " + deps[i]);
                to_compile.push({
                    name: name,
                    additional_args: additional_args,
                    includes: file_entry.include_paths
                });
                continue top; // No need to check the rest; we already know it is going
                // to be
                // compiled.
            }
        }
    }
}

var redo = false;

function check() {
    try {
        redo = false;
        for (var i = 0; i < files.length; i++) {
            check_comp(files[i]);
        }
        prune_out_similar();
    } catch (e) {
        if (e === "recompute")
            redo = true;
        else
            throw e;
    }
}

do {
    check();
} while (redo);

var errored = [];
var child_processes = [];
var processes_to_complete = to_compile.length;
var processes_completed = 0;

if (to_compile.length === 0) {
    if (!fs.existsSync(result)) {
        done_compiling();
    } else {
        console.log("No files to compile");
        // process.exit(0);
        done_compiling();
    }
}

for (var i = 0; i < to_compile.length; i++) {
    var name = to_compile[i].name;
    var cmdline = flags.join(" ") + " " + name + " " +
        to_compile[i].additional_args.join(" ") + " -c -o " +
        find_object_file_name(name);
    var in_args = cmdline.trim().split(/\s+/);
    for (var j = 0; j < to_compile[i].includes.length; j++) {
        in_args.push("-I" + to_compile[i].includes[j]);
    }

    console.log(cc + " " + in_args.join(" "));
    child_processes[i] = child_process.spawn(cc, in_args, {
        stdio: "inherit"
    });

    // Nasty hack
    child_processes[i].__filename__ = name;
    child_processes[i].on("close", function (code, signal) {
        if (code !== 0) {
            console.error("Failed to compile: " + this.__filename__);
            errored.push(this.__filename__);
        }
        processes_completed++;
        if (processes_completed === processes_to_complete) {
            done_compiling();
        }
    });
}

function done_compiling() {
    console.log(
        "" + processes_completed - errored.length + "/" + processes_to_complete +
        " compiled successfully!");
    console.log(
        "Files that errored: " + errored.length === 0 ? "NONE" :
            errored.join(", "));
    if (errored.length === 0) {
        // Final build step!
        // XXX: Fix for your platform
        if (fincc !== "ar")
            var cmdstring = fincc + " " + fincc_flags.join(" ") + flags.join(" ") + " -o " + result + " ";
        else
            var cmdstring = fincc + " rv -o " + result + " ";
        for (var i = 0; i < all_files.length; i++) {
            cmdstring += find_object_file_name(all_files[i]) + " ";
        }
        if (fincc !== "ar")
            cmdstring += end_flags.join(" ");
        // if (build_type !== "emscripten") { // apparently, emscripten has it built
        // in! how nice of them
        //    cmdstring += ["", "-lSDL", "-lSDLmain", "-I/usr/include/SDL"].join("
        //    ");
        //}
        try {
            console.log(cmdstring);
            child_process.execSync(cmdstring);
        } catch (e) {
            console.log("Compilation failed. See stdout for details");
            process.exit(1);
        }
    } else {
        process.exit(1);
    }
}
// A simple utility to get mtime.
function get_mtime(n) {
    var stat = fs.statSync(n);
    return stat.mtime.getTime();
}

function recompute_indiv(obj, file) {
    delete obj.__name; // Don't include in json
    for (var name in obj) {
        if (name == "description" || name == "archive") continue;
        var file_entry = obj[name];
        var cmd = "gcc -MM " + name + " "; // We do this the lazy way
        for (var i = 0; i < file_entry.include_paths.length; i++) {
            cmd += "-I" + file_entry.include_paths[i] + " ";
        }
        var output = child_process.execSync(cmd) + "";
        // First, connect the lines
        output = output.split("\\").join("");
        // Then get rid of that silly .o stuff at the beginning.
        output = output.substring(output.indexOf(":") + 1).trim();
        // Get our file list
        var files = output.split(/\s+/g);
        console.assert(
            name === files.shift(), "Error: incorrect file"); // The first one
        // should be the C
        // source itself
        file_entry.dependencies = files;
    }
    fs.writeFileSync(
        "build/" + file + "-files.json", JSON.stringify(obj, null, 4));
}
end_flags

function
    recompute_dependencies() {
    // Simply recompute all dependencies for our files
    for (var i = 0; i < files.length; i++) {
        recompute_indiv(files[i], files[i].__name);
    }
}

.ONESHELL:
.PHONY: docker

all: docker all-stow

docker: docker/Dockerfile_ubuntu
	 DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile_ubuntu .

docker/Dockerfile_ubuntu: makefiles/Makefile docker/Dockerfile_head_ubuntu docker/create_dockerfile.py
	mamba activate docker/conda_env
	python docker/create_dockerfile.py makefiles/Makefile docker/Dockerfile_head_ubuntu docker/Dockerfile_ubuntu
	mamba deactivate

progs=openssl libevent ncurses tmux mambaforge golang rust cmake \
	  git nvim pyenv pipx zotero buildg lua luarocks libgit2 exa stow \
	  bat broot ripgrep tealdeer zoxide du-dust fd-find git-delta \
	  bottom mcfly starship glow lazygit duf task chezmoi \
	  isort mypy black pyright

all-stow-targets:= $(foreach prog, ${progs}, stow-${prog})
all-unstow-targets:= $(foreach prog, ${progs}, unstow-${prog})

all-stow: $(all-stow-targets)
all-unstow: $(all-unstow-targets)

STOW_SRC=${HOME}/.stow
STOW_TGT=${HOME}/stowed
stow-%:
	mkdir -p ${STOW_TGT}/bin
	mkdir -p ${STOW_TGT}/lib
	mkdir -p ${STOW_TGT}/lib/pkgconfig
	for file in ${STOW_SRC}/$*/bin/*; do if [ -f $$file ]; then echo $$file; ln -s -f -T $$file ${STOW_TGT}/bin/$$(basename $$file); fi; done
	for file in ${STOW_SRC}/$*/lib/*; do if [ -f $$file ]; then echo $$file; ln -s -f -T $$file ${STOW_TGT}/lib/$$(basename $$file); fi; done
	for file in ${STOW_SRC}/$*/lib/pkgconfig/*; do if [ -f $$file ]; then echo $$file; ln -s -f -T $$file ${STOW_TGT}/lib/pkgconfig/$$(basename $$file); fi; done

unstow-%:
	for file in ${STOW_SRC}/$*/bin/*; do if [ -f $$file ]; then echo $$file; rm -f ${STOW_TGT}/bin/$$(basename $$file); fi; done
	for file in ${STOW_SRC}/$*/lib/*; do if [ -f $$file ]; then echo $$file; rm -f ${STOW_TGT}/lib/$$(basename $$file); fi; done
	for file in ${STOW_SRC}/$*/lib/pkgconfig/*; do if [ -f $$file ]; then echo $$file; rm -f ${STOW_TGT}/lib/pkgconfig/$$(basename $$file); fi; done
	
.ONESHELL:

PREFIX=${HOME}/.stow
export PREFIX

.PHONY: all clean uninstall
all clean uninstall:
	set -e
	$(MAKE) -C openssl $@
	$(MAKE) -C libevent $@
	$(MAKE) -C ncurses $@
	$(MAKE) -C tmux $@
	$(MAKE) -C mambaforge $@
	$(MAKE) -C golang $@
	$(MAKE) -C rust $@
	$(MAKE) -C cmake $@
	$(MAKE) -C git $@
	$(MAKE) -C neovim $@
	$(MAKE) -C pyenv $@
	$(MAKE) -C pipx $@
	$(MAKE) -C zotero $@
	$(MAKE) -C buildg $@
	$(MAKE) -C lua $@
	$(MAKE) -C luarocks $@
	$(MAKE) -C lmod $@
	$(MAKE) -C libgit2 $@
	$(MAKE) -C exa $@
	$(MAKE) -C stow $@
	# rust apps
	$(MAKE) -C rust_app APP=bat $@
	$(MAKE) -C rust_app APP=broot $@
	$(MAKE) -C rust_app APP=ripgrep APP_BINARY=rg $@
	$(MAKE) -C rust_app APP=tealdeer APP_BINARY=tldr $@
	$(MAKE) -C rust_app APP=zoxide $@
	$(MAKE) -C rust_app APP=du-dust APP_BINARY=dust $@
	$(MAKE) -C rust_app APP=fd-find APP_BINARY=fd $@
	$(MAKE) -C rust_app APP=git-delta APP_BINARY=delta $@
	$(MAKE) -C rust_app APP=bottom APP_BINARY=btm $@
	$(MAKE) -C rust_app APP=mcfly APP_BINARY=mcfly $@
	$(MAKE) -C rust_app APP=starship $@
	# go apps
	$(MAKE) -C go_app APP_URL=github.com/charmbracelet/glow APP_BINARY=glow $@
	$(MAKE) -C go_app APP_URL=github.com/muesli/duf APP_BINARY=duf $@
	$(MAKE) -C go_app APP_URL=github.com/go-task/task/v3/cmd/task APP_BINARY=task $@
	$(MAKE) -C go_app APP_URL=github.com/twpayne/chezmoi APP_BINARY=chezmoi $@
	# go app with custom module
	$(MAKE) -C lazygit $@
	# pipx apps
	$(MAKE) -C pipx_app APP=isort $@
	$(MAKE) -C pipx_app APP=mypy $@
	$(MAKE) -C pipx_app APP=black $@
	$(MAKE) -C pipx_app APP=pyright $@

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif

APP_PREFIX=${PREFIX}/lua
VERSION=5.4.4

.PHONY: all

all: ${APP_PREFIX}/bin/lua

lua-${VERSION}.tar.gz:
	wget https://www.lua.org/ftp/lua-${VERSION}.tar.gz


${APP_PREFIX}/bin/lua: lua-${VERSION}.tar.gz
	set -e
	tar xvf lua-${VERSION}.tar.gz
	cd lua-${VERSION}
	make all install INSTALL_TOP=${APP_PREFIX}

.PHONY: clean
clean:
	rm -f lua-${VERSION}.tar.gz
	rm -rf lua-${VERSION}

.PHONY: uninstall
uninstall:
	rm -rf ${APP_PREFIX}
	rm -f ${MODULE_PATH}

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif

APP_PREFIX=${PREFIX}/git
VERSION=2.38.0
MODULE_PATH=${PREFIX}/modules/home/git/default.lua
APP_TARFILE=git-${VERSION}.tar.gz

.PHONY: all

all: ${APP_PREFIX}/bin/git ${MODULE_PATH}

${APP_TARFILE}:
	wget https://www.kernel.org/pub/software/scm/git/${APP_TARFILE}

${APP_PREFIX}/bin/git: ${APP_TARFILE}
	tar zxvf ${APP_TARFILE}
	cd git-${VERSION}
	make configure
	./configure --prefix=${APP_PREFIX} --without-tcltk
	$(MAKE) -j && $(MAKE) install

${MODULE_PATH}: module_template
	mkdir -p $$(dirname ${MODULE_PATH})
	APP_PREFIX=${APP_PREFIX} envsubst '$$APP_PREFIX' < module_template > ${MODULE_PATH}

.PHONY: clean
clean:
	rm -rf git-${VERSION}
	rm -f ${APP_TARFILE}

.PHONY: uninstall
uninstall:
	rm -rf ${APP_PREFIX}
	rm -f ${MODULE_PATH}

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif

APP_PREFIX=${PREFIX}/cmake
VERSION=3.24.2
MODULE_PATH=${PREFIX}/modules/home/cmake/default.lua
APP_TARFILE=cmake-${VERSION}.tar.gz

.PHONY: all

all: ${APP_PREFIX}/bin/cmake ${MODULE_PATH}

${APP_TARFILE}:
	wget https://github.com/Kitware/CMake/releases/download/v${VERSION}/${APP_TARFILE}

.PHONY: deps
deps:
	$(MAKE) -C ../openssl

${APP_PREFIX}/bin/cmake: ${APP_TARFILE} | deps
	tar zxvf ${APP_TARFILE}
	cd cmake-${VERSION}
	PKG_CONFIG_PATH=${PREFIX}/openssl/lib/pkgconfig ./bootstrap --prefix=${APP_PREFIX}
	$(MAKE) -j && $(MAKE) install

${MODULE_PATH}: module_template
	mkdir -p $$(dirname ${MODULE_PATH})
	APP_PREFIX=${APP_PREFIX} envsubst '$$APP_PREFIX' < module_template > ${MODULE_PATH}

.PHONY: clean
clean:
	rm -rf cmake-${VERSION}
	rm -f ${APP_TARFILE}

.PHONY: uninstall
uninstall:
	rm -rf ${APP_PREFIX}
	rm -f ${MODULE_PATH}

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif

APP_PREFIX=${PREFIX}/go
VERSION=1.19.2
MODULE_PATH=${PREFIX}/modules/home/golang/default.lua
APP_TARFILE=go${VERSION}.linux-amd64.tar.gz

.PHONY: all
all: ${APP_PREFIX}/bin/go ${MODULE_PATH}

${APP_TARFILE}:
	wget https://go.dev/dl/${APP_TARFILE}


${APP_PREFIX}/bin/go: ${APP_TARFILE}
	tar zxvf ${APP_TARFILE} --directory ${PREFIX}
	touch ${APP_PREFIX}/bin/go

${MODULE_PATH}: module_template
	mkdir -p $$(dirname ${MODULE_PATH})
	APP_PREFIX=${APP_PREFIX} envsubst '$$APP_PREFIX' < module_template > ${MODULE_PATH}

.PHONY: clean
clean:
	rm -f ${APP_TARFILE}

.PHONY: uninstall
uninstall:
	rm -rf ${APP_PREFIX}
	rm -f ${MODULE_PATH}

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif

APP_PREFIX=${PREFIX}/pyenv
MODULE_PATH=${PREFIX}/modules/home/pyenv/default.lua
APP_DWNLOAD=pyenv-installer

.PHONY: all
all: ${APP_PREFIX}/bin/pyenv ${MODULE_PATH}

${APP_DWNLOAD}:
	wget --output-document ${APP_DWNLOAD} \
		https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/${APP_DWNLOAD}

.PHONY: deps
deps:
	$(MAKE) -C ../git

${APP_PREFIX}/bin/pyenv: ${APP_DWNLOAD} | deps
	export PATH=${PREFIX}/git/bin:${PATH}
	export PYENV_ROOT=${APP_PREFIX}
	chmod 770 ${APP_DWNLOAD}
	bash ${APP_DWNLOAD}
	mkdir -p ${APP_PREFIX}/shell
	${APP_PREFIX}/bin/pyenv init - > ${APP_PREFIX}/shell/init.sh

${MODULE_PATH}: module_template
	mkdir -p $$(dirname ${MODULE_PATH})
	APP_PREFIX=${APP_PREFIX} envsubst '$$APP_PREFIX' < module_template > ${MODULE_PATH}

.PHONY: clean
clean:
	rm -f ${APP_DWNLOAD}

.PHONY: uninstall
uninstall:
	rm -rf ${APP_PREFIX}
	rm -f ${MODULE_PATH}

.ONESHELL:

ifndef PREFIX
$(error PREFIX is not set)
endif
export PREFIX

ifndef APP_URL
$(error APP is not set)
endif

ifndef VERSION
VERSION=latest
endif

ifndef APP_BINARY
$(error APP_BINARY is not set)
endif

APP=$(shell basename ${APP_URL})
APP_PREFIX=${PREFIX}/${APP}
MODULE_PATH=${PREFIX}/modules/home/${APP}/default.lua
GOBIN=${APP_PREFIX}/bin
GO=${PREFIX}/go/bin/go
export GOBIN

.PHONY: all
all: ${GOBIN}/${APP_BINARY} ${MODULE_PATH}

.PHONY: deps
	$(MAKE) -C ../golang

${GOBIN}/${APP_BINARY}: | deps
	mkdir -p $$(dirname ${GOBIN})
	$(GO) install ${APP_URL}@${VERSION}

${MODULE_PATH}: module_template
	mkdir -p $$(dirname ${MODULE_PATH})
	APP_PREFIX=${APP_PREFIX} APP=${APP} envsubst '$$APP_PREFIX $$APP' < module_template > ${MODULE_PATH}

.PHONY: clean
clean:

.PHONY: uninstall
uninstall:
	rm -f ${GOBIN}/APP_BINARY
	rm -f ${MODULE_PATH}
	
gradient-aws/common.tf: common/common.tf
	cp common/common.tf gradient-aws/common.tf

gradient-metal/common.tf: common/common.tf
	cp common/common.tf gradient-metal/common.tf

gradient-metal-gc/common.tf: common/common.tf
	cp common/common.tf gradient-metal-gc/common.tf

gradient-ps-cloud/common.tf: common/common.tf
	cp common/common.tf gradient-ps-cloud/common.tf

.PHONY: all
all: gradient-ps-cloud/common.tf gradient-metal/common.tf gradient-aws/common.tf gradient-metal-gc/common.tf

SHELL=/usr/bin/env bash

GPU_PROGRAM           ?= nvidia-smi
GPU_RESULT             = $(shell which $(GPU_PROGRAM) 2>/dev/null)
GPU_TEST               = $(notdir $(GPU_RESULT))

ifeq ($(GPU_TEST), $(GPU_PROGRAM))
	GPU_FLAG           = --gpus all
endif

PYTORCH_DOWNLOAD_LINK ?= https://download.pytorch.org/whl/test/cu101/torch_test.html
# Used by CI to do plain buildkit progress
BUILD_PROGRESS        ?=
DOCKER_BUILD           = cat Dockerfile | DOCKER_BUILDKIT=1 docker build --target $@ $(BUILD_PROGRESS) --build-arg "PYTORCH_DOWNLOAD_LINK=$(PYTORCH_DOWNLOAD_LINK)" -t pytorch/integration-testing:$@ -
DOCKER_RUN             = set -o pipefail; docker run --rm -it $(GPU_FLAG) --shm-size 8G -v "$(PWD)/output:/output" pytorch/integration-testing:$@
CHOWN_TO_USER          = docker run --rm -v "$(PWD)":/v -w /v alpine chown -R "$(shell id -u):$(shell id -g)" .


.PHONY: all
all:
	@echo "please specify your target"

logs/:
	mkdir -p logs/

.PHONY: pyro
pyro: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest -v -c /dev/null -n auto --color=no --junitxml=/output/$@_results.xml --stage unit | tee logs/$@.log

.PHONY: detectron2
detectron2: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	# We have to install detectron2 here since it won't actually do the cuda build unless it has direct access to the GPU, :shrug:
	$(DOCKER_RUN) \
		sh -c 'pip install -U -e /detectron2 && pytest -v --color=no --junitxml=/output/$@_results.xml /detectron2/tests' 2>&1 \
			| tee logs/$@.log

.PHONY: transformers
transformers: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest -n auto --dist=loadfile --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>&1 | tee logs/$@.log

.PHONY: fairseq
fairseq: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest --color=no --junitxml=/output/$@_results.xml -s -v ./tests/ 2>/dev/null | tee logs/$@.log

.PHONY: pytorch-lightning
pytorch-lightning: logs/
	$(DOCKER_BUILD)
	$(DOCKER_RUN) pip list > logs/$@_metadata.log
	$(DOCKER_RUN) pytest pytorch_lightning tests -v --durations=0 --ignore=tests/loggers/test_all.py --ignore=tests/models/test_amp.py --junitxml=/output/$@_results.xml 2>/dev/null | tee logs/$@.log

.PHONY: chown-to-user
chown-to-user:
	$(CHOWN_TO_USER)

.PHONY: clean
clean: chown-to-user
	$(RM) -r output/
	$(RM) -r logs/
	
	
