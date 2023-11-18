# soil base scripts
soil hackathon

## R scripts to support the SOIL project.

### r_get_deps.R
Inspects all the R scripts in a development folder and finds lines calling library() function. Annotates all dependencies found in the r_deps.R file.
#### Rscript r_get_deps.R
Options:
  -h, --help      Show this help message and exit.
  -p, --path      Path to folder where to inspect for dependencies in R files. As a result, r_deps.R file will be created here.
  
### r_deps.R
lists and installs all R package dependencies in a development folder.
#### Rscript r_deps.R [options]
Options:
  -h, --help      Show this help message and exit.
  -dl, --deps     Print R dependencies of the project.

### r_togit.R
#### Rscript r_togit.R [options]
Options:
   -h, --help      Show this help message and exit.
   -g, --github    Open the GitHub repository for this project.
   -s, --script    Path to the R script to commit.
   -c, --commit    Commit message for the Git commit.
   -b, --branch    Branch to commit changes.
   -rg, --rgit     Remote Git repository link.
   -lg, --lgit     Local Git repository path.

### r_toconda.R
Creates a conda env .yaml using the R version and the R packages in the specific versions found in r_deps.R file.
#### Rscript r_get_deps.R
Options:
  -h, --help      Show this help message and exit.
  -p, --path      Path to r_deps.R file in the desired folder.

conda env create -f r_environment.yaml
conda activate r_environment
Rscript r_deps.R

### Dockerfile
This docker script will create a docker container using R version and R packages and versions specified in r_deps.R
### execute at path where the Dockerfile and r_deps.R are located.
docker build -t your-image-name .
docker run -it your-image-name

