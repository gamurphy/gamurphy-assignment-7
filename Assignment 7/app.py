from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
from scipy.stats import t


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    #Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)  # Replace with code to generate random values for X

    #Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N) #code to generate Y

    #Fit a linear regression model to X and Y
    #X_reshaped = X.reshape(-1, 1)  # Reshape X to be a 2D array
    X = X.reshape(-1, 1)
    model = LinearRegression() # Initialize the LinearRegression model
    model.fit(X,Y) #Fit the model to X and Y
    slope = model.coef_[0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_  # Extract the intercept from the fitted model

    #Scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    # Replace with code to generate and save the scatter plot
            #generating the scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(X,Y, color="blue", alpha=0.5, label="Data Points")
        #Generate points for the regression line
    x_values = np.linspace(0, 1, 100).reshape(-1, 1)
    y_values = slope * x_values + intercept
    plt.plot(x_values, y_values, color='red', label=f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
        # Label the plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Scatter Plot with Regression Line: Y = {slope:.2f}X + {intercept:.2f}")
    plt.legend()
        #code to generate and save the plot
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        #Simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)  # Replace with code to generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)   # Replace with code to generate simulated Y values

        #Fit linear regression to simulated data and store slope and intercept
        X_sim = X_sim.reshape(-1,1)
        sim_model = LinearRegression()  # Replace with code to fit the model
        sim_model.fit(X_sim,Y_sim)
        sim_slope = sim_model.coef_[0] # Extract slope from sim_model
        sim_intercept = sim_model.intercept_  # Extract intercept from sim_model

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    #Plotted histograms of slopes and intercepts
    #Code to generate and save the histogram plot
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    #Data needed for further analysis, including slopes and intercepts
    #Calculated proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(s > slope for s in slopes) / S #Code to calculate proportion of slopes more extreme than observed
    intercept_extreme = sum(i < intercept for i in intercepts) / S  # Replace with code to calculate proportion of intercepts more extreme than observed

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0


    # Calculate the difference between observed and hypothesized values
    observed_diff = observed_stat - hypothesized_value

    # TODO 10: Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_diff))
    else:
        return {"error": "Invalid test type. Choose 'not_equal_to', 'greater_than', or 'less_than'."}, 400

    #If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    fun_message = "Size Matters! Your p is not small :)"
    if p_value <=0.05:
        fun_message = "Size matters! Your p is small :("

    #Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Statistic")
    plt.axvline(hypothesized_value, color="green", linestyle="--", label="Hypothesized Value")
    plt.xlabel("Simulated Statistics")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Simulated {parameter.capitalize()}s")
    plt.legend()
    plt.savefig(plot3_path)# Save plot as an image file
    plt.close() 

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        #Uncomment the following lines when implemented
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    #Mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    #Calculate confidence interval for the parameter estimate
     #Determine z-critical value based on confidence level
    if confidence_level == 90:
        z_critical = 1.645
    elif confidence_level == 95:
        z_critical = 1.96
    elif confidence_level == 99:
        z_critical = 2.576
    else:
        raise ValueError("Unsupported confidence level. Choose 90, 95, or 99.")

    # Calculate confidence interval for the parameter estimate
    margin_of_error = z_critical * (std_estimate / np.sqrt(N))
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    #Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value


    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))    #Create the plot
    # Plot individual estimates as gray points
    plt.scatter(estimates, np.zeros_like(estimates), color="gray", alpha=0.6, label="Individual Estimates", marker='o')
    # Plot mean estimate as a colored point (green if includes true parameter, red if not)
    mean_color = "green" if includes_true else "red"
    plt.scatter(mean_estimate, 0, color=mean_color, s=100, label="Mean Estimate", zorder=5)
    # Plot the confidence interval as a horizontal line
    plt.hlines(0, ci_lower, ci_upper, color="blue", linewidth=2, label="Confidence Interval")
    # Plot the true parameter value as a vertical line
    plt.axvline(true_param, color="purple", linestyle="--", linewidth=2, label="True Parameter")
    # Add labels and legend
    plt.xlabel("Estimate Values")
    plt.yticks([])  # Hide y-axis as we are only interested in x-axis values
    plt.legend()
    plt.title("Estimates, Confidence Interval, and True Parameter")
    # Save the plot to a file
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
