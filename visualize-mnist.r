# TODO: Remove reactiveValues and use reactiveVal for unified naming
library(keras)
library(shiny)
library(ggrepel)
library(tidyverse)

# Load MNIST dataset, normalize and flatten 28x28 images using row-major layout (which is used by tensorflow).
# Also normalize the grayscale values into [0; 1] and one-hot-encode the labels.
mnist <- dataset_mnist()
MNIST <- list(x.train       = array_reshape(mnist$train$x, c(nrow(mnist$train$x), 784)) / 255,
              x.test        = array_reshape(mnist$test$x, c(nrow(mnist$test$x), 784)) / 255,
              y.train       = to_categorical(mnist$train$y, 10),
              y.test        = to_categorical(mnist$test$y, 10),
              y.train.digit = mnist$train$y, # keep the single-digit labels for convenience
              y.test.digit  = mnist$test$y)

# Return ggplot-heatmap from a row-major 1D vector.
# 
# Kept generic enough to plot arbitrary 2D heatmaps, since ggplot does not have any heatmap geoms and 'image()' rotates
# the image for no healthy reason. Might be reused at some point.
plot_heatmap <- function(data, nrow = 1, ncol = length(data)) {

  X = rep(1:nrow, times  = ncol)
  Y = rep(ncol:1, each = nrow)
  heat.map <- ggplot()                                    +
    geom_raster(aes(x = X, y = Y, fill = data))           +
    coord_cartesian(xlim = c(1, nrow), ylim = c(1, nrow)) +
    theme_void()                                          +
    theme(legend.position="none")

  return(heat.map)
}

# Return ggplot of training history. Expects a wide-format tibble consisting of all metrics to be displayed.
# 
# The x-axis displays the epoch (the integral index of each observation which need not be provided). The y-axis
# shows the numerical value of each metric. The metric type is mapped to the color aesthetic (one color per
# column in the input tibble).
plot_training_history <- function(data,             # wide-format tibble
                                  geom = geom_line) # any geom compatible with the x, y and color aesthetics
{
  if(nrow(data) == 0) {
    return(ggplot)
  }

  plot.data <- data %>%
    mutate(Epoch = row_number()) %>%
    gather(-Epoch, key = 'Metric', value = 'Value')

  history.plot <- ggplot(plot.data) +
    geom(aes(x = Epoch, y = Value, color = Metric))

  return(history.plot)
}

# Return indices of false predictions
#
# The model's predictions of 'x' are argmax'ed and compaired against y.
get_false_predictions <- function(model, x, y) {

  predicted.digits <- model %>% predict(x) %>% k_argmax %>% as.integer
  return(which(predicted.digits != y))
}

# Return untrained Multi-Layer Perceptron Keras model with a single hidden layer.
#
# The number of hidden neurons is parameterized.
initialize_model <- function(input.size = 784, hidden.units = 16, output.size = 10) {

  # Defining is a 2-part process: Firstly, the layers are specified. Secondly, the loss-function and the optimization
  # algorithm are chosed. The second step is referred to as 'compiling' the model in Keras lingo.
  model <- keras_model_sequential(name = "MNIST Classifier")                                                   %>%
    layer_dense(units = hidden.units, activation = 'relu', input_shape = c(input.size), name = "Hidden-Layer") %>%
    layer_dense(units = output.size, activation = 'softmax', name = "Output-Layer")                            %>%
    compile(loss = 'categorical_crossentropy', metrics = c('accuracy'), optimizer = optimizer_rmsprop())

  return(model)
}

# Returns ggplot barchart over estimates for each class 0 - 9
plot_digit_estimates <- function(estimates) {

  estimates <- as.vector(estimates)
  if(length(estimates) == 0) return(ggplot())

  bar.chart <- 
    ggplot() +
      geom_col(aes(x = 0:(length(estimates)-1), y = estimates, fill=as.factor(0:(length(estimates)-1))), show.legend = F) +
      scale_x_continuous(breaks = 0:9) +
      labs(title = "Class Estimates") +
      theme(
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title.x = element_blank(),
        plot.title = element_text(size=16, face="bold", hjust=0.5),
        axis.text.y = element_blank(),
        axis.text.x = element_text(face = "plain", size=12)
      )
  return(bar.chart)
}

### Define the app's user interface
ui <- fluidPage(title = "Live Neural Networks Demo", width="960px",
  fluidRow(
    column(4, offset = 2, align="center",
      h2("Test Set"),
      sliderInput('test.index', label=NULL, min=1, max=10000, value=1000),
      strong(textOutput(outputId = 'label.test')),
      plotOutput(outputId = 'test.image.plot')
    ),
    column(4, align = "center",
      h2("Training Set"),
      sliderInput('train.index', label=NULL, min=1, max=60000, value=8400),
      strong(textOutput(outputId = 'label.train')),
      plotOutput(outputId = 'training.image.plot')
    )
  )
#   fluidRow(
#     column(4, align="center",
#       sliderInput("epoch.increment", label="Numer of Epochs to Train", value=16, min=1, max=64, step=1),
#       splitLayout(cellWidths=c("65%", "35%"),
#         actionButton("reset.button", label="Reset + Run", width="100%"),
#         actionButton("run.button", label="Run", width="100%")
#       ),
#       br(),
#       splitLayout(
#         h5("Hidden neurons: ", style="padding-bottom: -10px;"),
#         numericInput("hl.size", label=NULL, min=1, max=128, step=1, value=16)),
#       # TODO: Properly concatenate the static prefix and the dynamic counter value
#       h4(textOutput("epoch.counter")),
#       br(),
#       radioButtons("geom.plot.option", label=NULL, choices=list("Point", "Line"), inline=T),
#       checkboxInput("show.train.acc.option", label="Show training set history", value=T)
#     ),
#     column(8, align="center",
#       plotOutput("training.history.plot")
#     )
#   ),
#   fluidRow(
#     column(4, align="center",
#       actionButton('init.button', label="Load Test Set Predictions", width = "100%"),
#       br(), br(),
#       uiOutput('false.prediction.slider'),
#       br(), br(),
#       splitLayout(
#         strong(textOutput('label')),
#         em(textOutput('prediction'))
#       ),
#       checkboxInput("show.correct.option", "Show Correct Predictions", T),
#       br(),
#       plotOutput('image.pixmap')
#     ),
#     column(6, offset=1, align="center",
#       plotOutput('estimates.barchart')
#     )
#   )
)

### Define the app's reactive outputs, i.e. the 'server'
server <- function(input, output) {

  # define reactive parameters and their dependent variables
  TRAINING.HISTORY <- reactiveVal(tibble())
  MODEL <- reactiveVal(initialize_model())
  EPOCH.COUNTER <- reactive({

    return(nrow(TRAINING.HISTORY()))
  }) # mirrors the height of the training dataframe


  FALSE.PREDICTIONS <- reactive({
    # BUG: dependency on the model is not captured (this reactive environment is not executed when learning and
    # FALSE.PREDICTIoNS and FALSE.PRE). Use an explicit dependency on TRAINING.HISTORY as a workaround for now.
    TRAINING.HISTORY()

    return(get_false_predictions(MODEL(), MNIST$x.test, MNIST$y.test.digit) )
  })

  FALSE.PREDICTIONS.INDEX <- reactive({

    return(FALSE.PREDICTIONS()[input$prediction.slider])
  })

  # Pixmap of test image
  output$test.image.plot <- renderPlot({

   return(plot_heatmap(MNIST$x.test[input$test.index,], nrow = 28, ncol = 28))
  })

  # Pixmap of training image
  output$training.image.plot <- renderPlot({

    return(plot_heatmap(MNIST$x.train[input$train.index,], nrow = 28, ncol = 28))
  })

  # Label of test image
  output$label.test <- renderText({

    return(paste0("Label: ", MNIST$y.test.digit[[input$test.index]]))
  })

  # Label of training image
  output$label.train <- renderText({

    return(paste0("Label: ", MNIST$y.train[[input$train.index]]))
  })

  # render epoch counter
  output$epoch.counter <- renderText({

    return(paste0("Epochs trained: ", EPOCH.COUNTER()))
  })

  # render training history plot
  output$training.history.plot <- renderPlot({

    return(plot_training_history(TRAINING.HISTORY()))
  })

  # 'reset.button' callback: Reset the model.
  observeEvent(input$reset.button, {

    MODEL(initialize_model())
    TRAINING.HISTORY(tibble())
  })

  # 'run.button' callback: Perform training.
  observeEvent(input$run.button, {

    fit.result <- fit(MODEL(), MNIST$x.train, MNIST$y.train,
                      initial_epoch = EPOCH.COUNTER(),
                      epochs = EPOCH.COUNTER() + input$epoch.increment,
                      validation_split = 0.2,
                      batch_size = 12000)

    # append newly gathered values. The column names are deduced automatically
    delta <- fit.result$metrics %>% as_tibble

    TRAINING.HISTORY(bind_rows(TRAINING.HISTORY(), delta))
  })

  output$prediction <- renderText({

    if(length(FALSE.PREDICTIONS.INDEX()) == 0) return("") # catch uninitialized value

    estimates <- MODEL() %>% predict(MNIST$x.test[FALSE.PREDICTIONS.INDEX(), ] %>% matrix(nrow=1))
    predicted.digit <- estimates %>% k_argmax %>% as.integer
    return(paste0("Prediction: ", predicted.digit))
  })

  output$label <- renderText({

    if(length(FALSE.PREDICTIONS.INDEX()) == 0) return("") # catch uninitialized value

    return(paste0("Label: ", MNIST$y.test.digit[FALSE.PREDICTIONS.INDEX()]))
  })

  output$false.prediction.slider <- renderUI({

    sliderInput('prediction.slider', label="False Prediction", min=1, max=length(FALSE.PREDICTIONS()), value=1)
  })
 
  # render estimates' barchart
  output$estimates.barchart <- renderPlot({

    if(length(FALSE.PREDICTIONS.INDEX()) == 0) return("") # catch uninitialized value

    estimates <- MODEL() %>% predict(MNIST$x.test[FALSE.PREDICTIONS.INDEX(), ] %>% matrix(nrow=1))
    return(plot_digit_estimates(estimates))
  })

  output$image.pixmap <- renderPlot({

    if(length(FALSE.PREDICTIONS.INDEX()) == 0) return("") # catch uninitialized value

    return(plot_heatmap(MNIST$x.test[FALSE.PREDICTIONS.INDEX(),], nrow = 28, ncol = 28))
  })
}


# Run the app
app <- shinyApp(ui = ui, server = server)
runApp(app)
