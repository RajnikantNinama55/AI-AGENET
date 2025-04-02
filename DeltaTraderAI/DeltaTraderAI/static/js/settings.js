/**
 * Settings.js - Settings page functionality for the trading bot
 * Handles API configuration and trading settings forms
 */

$(document).ready(function() {
    // Load current settings from the API
    loadCurrentSettings();
    
    // Toggle API secret visibility
    $('#toggle-secret').click(function() {
        const secretInput = $('#api-secret');
        const type = secretInput.attr('type') === 'password' ? 'text' : 'password';
        secretInput.attr('type', type);
        
        // Toggle icon
        const icon = $(this).find('i');
        if (type === 'text') {
            icon.removeClass('fa-eye').addClass('fa-eye-slash');
        } else {
            icon.removeClass('fa-eye-slash').addClass('fa-eye');
        }
    });
    
    // Handle API settings form submission
    $('#api-settings-form').submit(function(e) {
        e.preventDefault();
        
        const apiKey = $('#api-key').val();
        const apiSecret = $('#api-secret').val();
        
        // Save API settings
        $.ajax({
            url: '/api/save_api_settings',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                api_key: apiKey,
                api_secret: apiSecret
            }),
            success: function(response) {
                if (response.success) {
                    showToast('Success', 'API settings saved successfully', 'success');
                } else {
                    showToast('Error', response.message || 'Failed to save API settings', 'error');
                }
            },
            error: function(xhr) {
                showToast('Error', 'Failed to save API settings: Network error', 'error');
            }
        });
    });
    
    // Handle trading settings form submission
    $('#trading-settings-form').submit(function(e) {
        e.preventDefault();
        
        const riskPerTrade = $('#risk-per-trade').val();
        const maxTrades = $('#max-trades').val();
        const tradeTimeout = $('#trade-timeout').val();
        
        // Save trading settings
        $.ajax({
            url: '/api/save_trading_settings',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                risk_per_trade: parseFloat(riskPerTrade),
                max_trades: parseInt(maxTrades),
                trade_timeout: parseInt(tradeTimeout)
            }),
            success: function(response) {
                if (response.success) {
                    showToast('Success', 'Trading settings saved successfully', 'success');
                } else {
                    showToast('Error', response.message || 'Failed to save trading settings', 'error');
                }
            },
            error: function(xhr) {
                showToast('Error', 'Failed to save trading settings: Network error', 'error');
            }
        });
    });
    
    // Handle strategy form submissions
    $('.strategy-form').submit(function(e) {
        e.preventDefault();
        
        const strategyType = $(this).data('strategy');
        const isActive = $(this).find('.strategy-active').is(':checked');
        
        // Collect parameters
        const parameters = {};
        $(this).find('input[name]').each(function() {
            const name = $(this).attr('name');
            const value = $(this).attr('type') === 'number' ? parseFloat($(this).val()) : $(this).val();
            parameters[name] = value;
        });
        
        // Save strategy settings
        $.ajax({
            url: '/api/save_strategy_settings',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                strategy_type: strategyType,
                is_active: isActive,
                parameters: parameters
            }),
            success: function(response) {
                if (response.success) {
                    showToast('Success', `${strategyType.replace('_', ' ')} strategy settings saved`, 'success');
                } else {
                    showToast('Error', response.message || 'Failed to save strategy settings', 'error');
                }
            },
            error: function(xhr) {
                showToast('Error', 'Failed to save strategy settings: Network error', 'error');
            }
        });
    });
});

/**
 * Load current settings from the API
 */
function loadCurrentSettings() {
    $.ajax({
        url: '/api/get_user_settings',
        type: 'GET',
        success: function(response) {
            if (response.success) {
                // Populate API settings
                $('#api-key').val(response.api_key || '');
                
                // For security, we don't load the API secret
                // Just indicate if one exists
                if (response.api_secret) {
                    $('#api-secret').attr('placeholder', '••••••••••••••••••••••••••••••');
                }
                
                // Populate trading settings
                const tradingSettings = response.trading_settings || {};
                $('#risk-per-trade').val(tradingSettings.risk_per_trade || 1.0);
                $('#max-trades').val(tradingSettings.max_trades || 3);
                $('#trade-timeout').val(tradingSettings.trade_timeout || 48);
                
                // Populate strategy settings
                // This would load them if we had them in the response
            } else {
                showToast('Error', response.message || 'Failed to load settings', 'error');
            }
        },
        error: function(xhr) {
            showToast('Error', 'Failed to load settings: Network error', 'error');
        }
    });
}

/**
 * Show a toast notification
 * @param {string} title - Toast title
 * @param {string} message - Toast message
 * @param {string} type - Toast type (success, error, warning, info)
 */
function showToast(title, message, type = 'info') {
    // Create toast container if it doesn't exist
    if ($('#toast-container').length === 0) {
        $('body').append('<div id="toast-container" class="position-fixed top-0 end-0 p-3" style="z-index: 1050;"></div>');
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'} text-white">
                <strong class="me-auto">${title}</strong>
                <small>Now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    // Add toast to container
    $('#toast-container').append(toast);
    
    // Initialize and show the toast
    const toastElement = new bootstrap.Toast(document.getElementById(toastId), {
        autohide: true,
        delay: 5000
    });
    toastElement.show();
}