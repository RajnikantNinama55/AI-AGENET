/**
 * Charts.js - Utility functions for chart creation and management
 * Delta Exchange Trading Bot
 */

// Chart color palette
const chartColors = {
    primary: '#2196F3',  // Chart blue
    secondary: '#144272', // Secondary blue
    success: '#25D366',  // Green
    danger: '#FF4444',   // Red
    warning: '#FFC107',  // Amber
    info: '#00BCD4',     // Cyan
    background: '#000000', // Black
    text: '#FFFFFF',     // White
    gridLines: 'rgba(255, 255, 255, 0.1)', // Grid lines
    tooltipBackground: 'rgba(15, 23, 42, 0.9)' // Tooltip background
};

// Set global Chart.js defaults
Chart.defaults.color = chartColors.text;
Chart.defaults.borderColor = chartColors.gridLines;
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";

// Custom tooltip styling
Chart.defaults.plugins.tooltip.backgroundColor = chartColors.tooltipBackground;
Chart.defaults.plugins.tooltip.titleColor = chartColors.text;
Chart.defaults.plugins.tooltip.bodyColor = chartColors.text;
Chart.defaults.plugins.tooltip.borderColor = chartColors.gridLines;
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 6;

/**
 * Create a performance chart to show portfolio equity curve
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} dates - Array of date strings
 * @param {Array} values - Array of portfolio values
 * @param {Object} options - Additional chart options
 * @returns {Chart} The created chart
 */
function createPerformanceChart(canvasId, dates, values, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Calculate percentage change
    const initialValue = values[0];
    const pctChange = values.map(value => ((value - initialValue) / initialValue) * 100);
    
    // Calculate annotations (if needed)
    const annotations = {};
    if (options.showDrawdown && options.maxDrawdown) {
        annotations.maxDrawdown = {
            type: 'line',
            scaleID: 'y',
            value: -options.maxDrawdown,
            borderColor: chartColors.danger,
            borderWidth: 1,
            borderDash: [5, 5],
            label: {
                content: `Max Drawdown: ${options.maxDrawdown.toFixed(2)}%`,
                enabled: true,
                position: 'start',
                backgroundColor: chartColors.danger
            }
        };
    }
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Portfolio Performance',
                data: options.showPercent ? pctChange : values,
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHitRadius: 10,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (options.showPercent) {
                                return `Performance: ${context.raw.toFixed(2)}%`;
                            } else {
                                return `Balance: $${context.raw.toFixed(2)}`;
                            }
                        }
                    }
                },
                annotation: {
                    annotations: annotations
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 10,
                        maxRotation: 0
                    }
                },
                y: {
                    grid: {
                        color: chartColors.gridLines
                    },
                    ticks: {
                        callback: function(value) {
                            if (options.showPercent) {
                                return value.toFixed(1) + '%';
                            } else {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create a chart comparing multiple strategies
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} labels - Strategy names
 * @param {Array} datasets - Array of datasets for each metric
 * @returns {Chart} The created chart
 */
function createStrategyComparisonChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets.map((dataset, index) => {
                const colors = [
                    chartColors.primary,
                    chartColors.success,
                    chartColors.warning,
                    chartColors.info
                ];
                
                return {
                    label: dataset.label,
                    data: dataset.data,
                    backgroundColor: colors[index % colors.length],
                    borderColor: 'transparent',
                    borderWidth: 1,
                    borderRadius: 4
                };
            })
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 15,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2);
                                if (context.dataset.label.includes('Return') || context.dataset.label.includes('Win Rate')) {
                                    label += '%';
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: chartColors.gridLines
                    },
                    ticks: {
                        beginAtZero: true
                    }
                }
            }
        }
    });
}

/**
 * Create a pie chart for strategy allocation
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} labels - Strategy names
 * @param {Array} values - Strategy weights
 * @returns {Chart} The created chart
 */
function createAllocationChart(canvasId, labels, values) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const colors = [
        chartColors.primary,  // Blue
        chartColors.success,  // Green
        chartColors.info,     // Cyan
        chartColors.warning   // Amber
    ];
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: chartColors.background,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((acc, data) => acc + data, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${percentage}%`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create a candlestick chart (for price data)
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} dates - Array of date strings
 * @param {Array} ohlc - Array of objects with open, high, low, close values
 * @param {Array} indicators - Optional array of indicator datasets
 * @returns {Chart} The created chart
 */
function createCandlestickChart(canvasId, dates, ohlc, indicators = []) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Prepare data for candlestick chart
    const candleData = ohlc.map((item, index) => ({
        x: dates[index],
        o: item.open,
        h: item.high,
        l: item.low,
        c: item.close
    }));
    
    // Create datasets for candlesticks
    const datasets = [{
        label: 'Price',
        data: candleData,
        color: {
            up: chartColors.success,
            down: chartColors.danger,
            unchanged: chartColors.text
        }
    }];
    
    // Add indicator datasets
    indicators.forEach(indicator => {
        datasets.push({
            label: indicator.label,
            data: indicator.data.map((value, index) => ({
                x: dates[index],
                y: value
            })),
            type: 'line',
            borderColor: indicator.color || chartColors.info,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
        });
    });
    
    return new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    position: 'right'
                }
            },
            plugins: {
                legend: {
                    labels: {
                        filter: function(legendItem) {
                            // Filter out the candlestick dataset from the legend
                            return legendItem.text !== 'Price';
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.type === 'candlestick') {
                                const data = context.raw;
                                return [
                                    `Open: $${data.o.toFixed(2)}`,
                                    `High: $${data.h.toFixed(2)}`,
                                    `Low: $${data.l.toFixed(2)}`,
                                    `Close: $${data.c.toFixed(2)}`
                                ];
                            } else {
                                return `${context.dataset.label}: $${context.raw.y.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        }
    });
}

/**
 * Format date for charts based on timeframe
 * @param {string} dateString - Date string or timestamp
 * @param {string} timeframe - Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
 * @returns {string} Formatted date string
 */
function formatChartDate(dateString, timeframe) {
    const date = new Date(dateString);
    
    if (timeframe === '1d') {
        return date.toLocaleDateString();
    } else if (timeframe === '4h' || timeframe === '1h') {
        return `${date.toLocaleDateString()} ${date.getHours()}:00`;
    } else {
        return `${date.toLocaleDateString()} ${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
    }
}

/**
 * Update data in an existing chart
 * @param {Chart} chart - Chart.js instance
 * @param {Array} labels - New labels
 * @param {Array} values - New values
 */
function updateChartData(chart, labels, values) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
}
