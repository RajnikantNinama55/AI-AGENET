/**
 * Dashboard.js - Dashboard functionality for the trading bot
 * Handles charts, data loading, and interaction for the dashboard page
 */

$(document).ready(function() {
    // Initialize variables for charts
    let performanceChart = null;
    let strategyComparisonChart = null;
    let allocationChart = null;
    let currentPage = 1;
    let totalPages = 1;
    let tradesPerPage = 10;
    
    // Initialize dashboard with 1-month performance data
    initializeDashboard('1m');
    
    // Timeframe selection for performance chart
    $('.timeframe-btn').click(function() {
        $('.timeframe-btn').removeClass('active');
        $(this).addClass('active');
        
        const period = $(this).data('period');
        loadPerformanceData(period);
    });
    
    // Strategy and timeframe selection for comparison chart
    $('#strategy-symbol, #strategy-timeframe').change(function() {
        loadStrategyComparisonData();
    });
    
    // Trade history pagination
    $('#prev-page').click(function() {
        if (currentPage > 1) {
            currentPage--;
            loadTradeHistory(currentPage);
        }
    });
    
    $('#next-page').click(function() {
        if (currentPage < totalPages) {
            currentPage++;
            loadTradeHistory(currentPage);
        }
    });
    
    // Export trades button
    $('#export-trades').click(function() {
        exportTradeHistory();
    });
    
    /**
     * Initialize the dashboard with data
     * @param {string} period - Time period to load (1d, 1w, 1m, 3m, all)
     */
    function initializeDashboard(period) {
        loadSummaryMetrics();
        loadPerformanceData(period);
        loadStrategyComparisonData();
        loadAllocationData();
        loadTradeHistory(1);
        
        // Refresh data periodically
        setInterval(loadSummaryMetrics, 60000); // Every minute
        setInterval(function() {
            const activePeriod = $('.timeframe-btn.active').data('period');
            loadPerformanceData(activePeriod);
        }, 60000); // Every minute
    }
    
    /**
     * Load summary metrics for the dashboard
     */
    function loadSummaryMetrics() {
        $.get('/api/get_summary_metrics', function(data) {
            if (data.success) {
                // Update balance info
                $('#total-balance').text('$' + data.balance.toFixed(2));
                
                const balanceChange = data.balance_change;
                $('#balance-change').text(balanceChange.toFixed(2) + '%');
                
                if (balanceChange >= 0) {
                    $('#balance-change').removeClass('text-danger').addClass('text-success');
                } else {
                    $('#balance-change').removeClass('text-success').addClass('text-danger');
                }
                
                // Update win rate and trade info
                $('#win-rate').text(data.win_rate.toFixed(2) + '%');
                $('#total-trades').text(data.total_trades);
                
                // Update average trade
                $('#avg-trade').text('$' + data.avg_trade.toFixed(2));
                $('#profit-factor').text(data.profit_factor.toFixed(2));
                
                // Update drawdown info
                $('#max-drawdown').text(data.max_drawdown.toFixed(2) + '%');
                $('#recovery-time').text(data.recovery_time || 'N/A');
            }
        });
    }
    
    /**
     * Load performance data for the selected period
     * @param {string} period - Time period to load (1d, 1w, 1m, 3m, all)
     */
    function loadPerformanceData(period) {
        $.get('/api/get_performance_data', { period: period }, function(data) {
            if (data.success) {
                renderPerformanceChart(data.dates, data.values);
            }
        });
    }
    
    /**
     * Render the performance chart with the provided data
     * @param {Array} dates - Array of date strings
     * @param {Array} values - Array of portfolio values
     */
    function renderPerformanceChart(dates, values) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (performanceChart) {
            performanceChart.destroy();
        }
        
        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Portfolio Value',
                    data: values,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    fill: true,
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
                                return `Balance: $${context.raw.toFixed(2)}`;
                            }
                        }
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
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Load strategy comparison data
     */
    function loadStrategyComparisonData() {
        const symbol = $('#strategy-symbol').val();
        const timeframe = $('#strategy-timeframe').val();
        
        $.get('/api/get_strategy_comparison', { 
            symbol: symbol, 
            timeframe: timeframe 
        }, function(data) {
            if (data.success) {
                renderStrategyComparisonChart(data.strategies, data.returns, data.win_rates);
            }
        });
    }
    
    /**
     * Render the strategy comparison chart
     * @param {Array} strategies - Strategy names
     * @param {Array} returns - Strategy returns
     * @param {Array} winRates - Strategy win rates
     */
    function renderStrategyComparisonChart(strategies, returns, winRates) {
        const ctx = document.getElementById('strategyComparisonChart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (strategyComparisonChart) {
            strategyComparisonChart.destroy();
        }
        
        strategyComparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: strategies,
                datasets: [
                    {
                        label: 'Return (%)',
                        data: returns,
                        backgroundColor: 'rgba(33, 150, 243, 0.7)',
                        borderColor: 'transparent',
                        borderWidth: 1,
                        borderRadius: 4,
                        order: 1
                    },
                    {
                        label: 'Win Rate (%)',
                        data: winRates,
                        backgroundColor: 'rgba(37, 211, 102, 0.7)',
                        borderColor: 'transparent',
                        borderWidth: 1,
                        borderRadius: 4,
                        order: 2
                    }
                ]
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
                                    label += context.parsed.y.toFixed(2) + '%';
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
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Load strategy allocation data
     */
    function loadAllocationData() {
        $.get('/api/get_strategy_allocation', function(data) {
            if (data.success) {
                renderAllocationChart(data.strategies, data.weights);
                
                // Update the weight progress bars
                data.strategies.forEach((strategy, index) => {
                    const weight = data.weights[index];
                    const percent = (weight * 100).toFixed(0);
                    
                    switch(strategy) {
                        case 'Trend Following':
                            $('#trend-weight').text(percent + '%');
                            $('#trend-progress').css('width', percent + '%');
                            break;
                        case 'Mean Reversion':
                            $('#mean-weight').text(percent + '%');
                            $('#mean-progress').css('width', percent + '%');
                            break;
                        case 'Breakout':
                            $('#breakout-weight').text(percent + '%');
                            $('#breakout-progress').css('width', percent + '%');
                            break;
                        case 'Momentum':
                            $('#momentum-weight').text(percent + '%');
                            $('#momentum-progress').css('width', percent + '%');
                            break;
                    }
                });
            }
        });
    }
    
    /**
     * Render the allocation chart
     * @param {Array} strategies - Strategy names
     * @param {Array} weights - Strategy weights
     */
    function renderAllocationChart(strategies, weights) {
        const ctx = document.getElementById('allocationChart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (allocationChart) {
            allocationChart.destroy();
        }
        
        // Convert weights to percentages
        const percentages = weights.map(weight => weight * 100);
        
        allocationChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: strategies,
                datasets: [{
                    data: percentages,
                    backgroundColor: [
                        'rgba(33, 150, 243, 0.7)',   // Primary (blue)
                        'rgba(37, 211, 102, 0.7)',   // Success (green)
                        'rgba(0, 188, 212, 0.7)',    // Info (cyan)
                        'rgba(255, 193, 7, 0.7)'     // Warning (amber)
                    ],
                    borderColor: '#000000',
                    borderWidth: 1
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
                            boxWidth: 12,
                            padding: 10,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Load trade history data with pagination
     * @param {number} page - Page number to load
     */
    function loadTradeHistory(page) {
        $.get('/api/get_trade_history', { 
            page: page,
            per_page: tradesPerPage
        }, function(data) {
            if (data.success) {
                renderTradeHistory(data.trades, data.current_page, data.total_pages, data.total_trades);
            }
        });
    }
    
    /**
     * Render trade history table
     * @param {Array} trades - Array of trade objects
     * @param {number} currentPage - Current page number
     * @param {number} totalPages - Total number of pages
     * @param {number} totalTrades - Total number of trades
     */
    function renderTradeHistory(trades, currentPage, totalPages, totalTrades) {
        const tableBody = $('#trade-history');
        tableBody.empty();
        
        if (trades.length === 0) {
            tableBody.html('<tr><td colspan="11" class="text-center">No trades found</td></tr>');
            return;
        }
        
        trades.forEach(trade => {
            const pnlClass = parseFloat(trade.pnl) >= 0 ? 'text-success' : 'text-danger';
            const pnlPctClass = parseFloat(trade.pnl_percentage) >= 0 ? 'text-success' : 'text-danger';
            
            const entryTime = new Date(trade.entry_time).toLocaleString();
            const exitTime = trade.exit_time ? new Date(trade.exit_time).toLocaleString() : 'Open';
            
            tableBody.append(`
                <tr>
                    <td>${trade.id}</td>
                    <td>${entryTime}</td>
                    <td>${exitTime}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.strategy}</td>
                    <td>${trade.direction}</td>
                    <td>$${parseFloat(trade.entry_price).toFixed(2)}</td>
                    <td>${trade.exit_price ? '$' + parseFloat(trade.exit_price).toFixed(2) : '-'}</td>
                    <td>${parseFloat(trade.quantity).toFixed(6)}</td>
                    <td class="${pnlClass}">${trade.pnl ? '$' + parseFloat(trade.pnl).toFixed(2) : '-'}</td>
                    <td class="${pnlPctClass}">${trade.pnl_percentage ? parseFloat(trade.pnl_percentage).toFixed(2) + '%' : '-'}</td>
                </tr>
            `);
        });
        
        // Update pagination info
        $('#showing-trades').text(trades.length);
        $('#total-trade-count').text(totalTrades);
        $('#page-indicator').text(`Page ${currentPage} of ${totalPages}`);
        
        // Update pagination buttons
        $('#prev-page').prop('disabled', currentPage <= 1);
        $('#next-page').prop('disabled', currentPage >= totalPages);
        
        // Store current pagination state
        currentPage = currentPage;
        totalPages = totalPages;
    }
    
    /**
     * Export trade history to CSV
     */
    function exportTradeHistory() {
        $.get('/api/export_trade_history', function(data) {
            if (data.success) {
                // Create CSV content
                let csvContent = 'data:text/csv;charset=utf-8,';
                csvContent += 'ID,Entry Time,Exit Time,Symbol,Strategy,Direction,Entry Price,Exit Price,Quantity,PnL,PnL %\n';
                
                data.trades.forEach(trade => {
                    const entryTime = new Date(trade.entry_time).toLocaleString();
                    const exitTime = trade.exit_time ? new Date(trade.exit_time).toLocaleString() : 'Open';
                    
                    csvContent += `${trade.id},${entryTime},${exitTime},${trade.symbol},${trade.strategy},${trade.direction},$${parseFloat(trade.entry_price).toFixed(2)},`;
                    csvContent += `${trade.exit_price ? '$' + parseFloat(trade.exit_price).toFixed(2) : '-'},${parseFloat(trade.quantity).toFixed(6)},`;
                    csvContent += `${trade.pnl ? '$' + parseFloat(trade.pnl).toFixed(2) : '-'},${trade.pnl_percentage ? parseFloat(trade.pnl_percentage).toFixed(2) + '%' : '-'}\n`;
                });
                
                // Create download link
                const encodedUri = encodeURI(csvContent);
                const link = document.createElement('a');
                link.setAttribute('href', encodedUri);
                link.setAttribute('download', `trade_history_${new Date().toISOString().slice(0, 10)}.csv`);
                document.body.appendChild(link);
                
                // Trigger download
                link.click();
                
                // Clean up
                document.body.removeChild(link);
            } else {
                alert('Failed to export trade history.');
            }
        });
    }
});
