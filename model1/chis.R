### MULTINOMIAL FISHER


# data --------------------------------------------------------------------
glimpse(fn_luc18)
glimpse(fn_luc15)
glimpse(fn_luc09)
View(fn_luc18)


# charact to factor -------------------------------------------------------

# Identify character or factor variables
categorical_variables_18 <- sapply(fn_luc18, function(x) is.character(x) | is.factor(x))
categorical_data_18 <- fn_luc18[, categorical_variables_18]

categorical_variables_15 <- sapply(fn_luc15, function(x) is.character(x) | is.factor(x))
categorical_data_15 <- fn_luc18[, categorical_variables_15]

categorical_variables_09 <- sapply(fn_luc09, function(x) is.character(x) | is.factor(x))
categorical_data_09 <- fn_luc18[, categorical_variables_09]

# Loop through each variable
sink("e_analysis/categoricals_18.txt")
for (var in names(categorical_data_18)) {
  cat(paste("Contingency table for", var, ":\n"))
  
  # Create the contingency table
  contingency_table <- table(categorical_data_18[[var]])
  
  # Display the contingency table
  print(contingency_table)
  cat("\n")
}
sink()

# Loop through each variable
sink("e_analysis/categoricals_15.txt")
for (var in names(categorical_data_15)) {
  cat(paste("Contingency table for", var, ":\n"))
  
  # Create the contingency table
  contingency_table <- table(categorical_data_15[[var]])
  
  # Display the contingency table
  print(contingency_table)
  cat("\n")
}
sink()

# Loop through each variable
sink("e_analysis/categoricals_09.txt")
for (var in names(categorical_data_09)) {
  cat(paste("Contingency table for", var, ":\n"))
  
  # Create the contingency table
  contingency_table <- table(categorical_data_09[[var]])
  
  # Display the contingency table
  print(contingency_table)
  cat("\n")
}
sink()



# factors -------------------------------------------------------------
# Identify character or factor variables
fct_luc18 <- fn_luc18[, categorical_variables_18]
fct_luc15 <- fn_luc15[, categorical_variables_15]
fct_luc09 <- fn_luc09[, categorical_variables_09]

fct_luc18 <- select(fct_luc18, -c("LU1_recoded"))
fct_luc15 <- select(fct_luc15, -c("LU1_recoded"))
fct_luc09 <- select(fct_luc09, -c("LU1_recoded"))

glimpse(fct_luc18)

# Filter out the response variable from the list of categorical variables
resp_var <- "LU1_recoded"

# fct_luc18 <- fct_luc18[!names(fct_luc18) == resp_var]
# fct_luc15 <- fct_luc15[!names(fct_luc15) == resp_var]
# fct_luc09 <- fct_luc09[!names(fct_luc09) == resp_var]

# Perform chi-squared test for each combination
sink("e_analysis/chi_categoricals_pvalues_18.txt")
for (var in names(fct_luc18)) {
  contingency_table <- table(fn_luc18[[resp_var]], fn_luc18[[var]])
  result <- chisq.test(contingency_table)
  # Print variable name and formatted p-value
  if (result$p.value < 2.2e-16) {
    cat(paste("Chi-squared test for", var, ": p-value <", format(2.2e-16, digits = 4), "\n"))
  } else {
    cat(paste("Chi-squared test for", var, ": p-value =", format(result$p.value, digits = 4), "\n"))
  }
}
sink()


# Perform chi-squared test for each combination
sink("e_analysis/chi_categoricals_pvalues_15.txt")
for (var in names(fct_luc15)) {
  contingency_table <- table(fn_luc15[[resp_var]], fn_luc15[[var]])
  result <- chisq.test(contingency_table)
  # Print variable name and formatted p-value
  if (result$p.value < 2.2e-16) {
    cat(paste("Chi-squared test for", var, ": p-value <", format(2.2e-16, digits = 4), "\n"))
  } else {
    cat(paste("Chi-squared test for", var, ": p-value =", format(result$p.value, digits = 4), "\n"))
  }
}
sink()


# Perform chi-squared test for each combination
sink("e_analysis/chi_categoricals_pvalues_09.txt")
for (var in names(fct_luc09)) {
  contingency_table <- table(fn_luc09[[resp_var]], fn_luc09[[var]])
  result <- chisq.test(contingency_table)
  # Print variable name and formatted p-value
  if (result$p.value < 2.2e-16) {
    cat(paste("Chi-squared test for", var, ": p-value <", format(2.2e-16, digits = 4), "\n"))
  } else {
    cat(paste("Chi-squared test for", var, ": p-value =", format(result$p.value, digits = 4), "\n"))
  }
}
sink()




