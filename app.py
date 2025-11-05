import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Comparison Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Green AI theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom, #f0f9f4 0%, #e8f5e9 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(16,185,129,0.1);
        border-left: 4px solid #10b981;
    }
    h1 {
        color: #065f46;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        color: #047857;
    }
    .metric-card {
        background: linear-gradient(135deg, #10b981 0%, #0891b2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(16,185,129,0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }
</style>
""", unsafe_allow_html=True)

# Data loading function with caching
@st.cache_data
def load_data(file_path='runs_tasks_01_30.xlsx'):
    """Load and prepare data from Excel file"""
    try:
        # Try to load from file in same directory
        df = pd.read_excel(file_path)
        
        # Map Excel column names to expected column names
        column_mapping = {
            'task_id': 'Task_ID',
            'task_title': 'Task_Description', 
            'task_category': 'Category',
            'model_name': 'Model_Name',
            'quality_score_1to5': 'Quality_Score',
            'latency_sec': 'Latency_sec',
            'energy_kwh': 'Energy_kWh',
            'co2_kg': 'CO2_kg'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Calculate estimated cost based on energy consumption (simple approximation)
        # Assuming 0.20 EUR per kWh (approximate European electricity price)
        df['Cost_EUR'] = df['Energy_kWh'] * 0.20
        
        st.sidebar.success(f"Data loaded: {file_path}")
    except FileNotFoundError:
        st.sidebar.warning("File not found, using sample data")
        # Sample data structure matching TP1 requirements
        np.random.seed(42)
        models = ['LLaMA 3.1 8B', 'Gemma 8B', 'Mistral Small', 
                  'GPT-OSS 20B', 'GPT-5', 'DeepSeek R1']
        tasks = [f"Task {i}" for i in range(1, 31)]
        
        data = []
        for task_id, task in enumerate(tasks, 1):
            # Categorize tasks
            if task_id <= 10:
                category = "Factual & Rewriting"
            elif task_id <= 15:
                category = "Reasoning & Quantitative"
            elif task_id <= 20:
                category = "Programming & Debugging"
            elif task_id <= 25:
                category = "Knowledge & Reasoning"
            else:
                category = "Advanced & Creative"
            
            for model in models:
                # Simulate realistic values based on model size
                if '8B' in model:
                    base_quality = 3.5 if task_id <= 20 else 2.8
                    base_latency = 0.5
                    base_energy = 0.002
                    base_cost = 0.0005
                elif '20B' in model or 'Small' in model:
                    base_quality = 4.0 if task_id <= 25 else 3.5
                    base_latency = 1.2
                    base_energy = 0.005
                    base_cost = 0.002
                else:  # Large models
                    base_quality = 4.5 if task_id <= 30 else 4.2
                    base_latency = 2.5
                    base_energy = 0.015
                    base_cost = 0.008
                
                # Add randomness
                data.append({
                    'Task_ID': task_id,
                    'Task_Description': task,
                    'Category': category,
                    'Model_Name': model,
                    'Quality_Score': min(5, max(1, base_quality + np.random.normal(0, 0.3))),
                    'Latency_sec': max(0.1, base_latency + np.random.normal(0, 0.2)),
                    'Energy_kWh': max(0.001, base_energy + np.random.normal(0, 0.002)),
                    'CO2_kg': max(0.0001, (base_energy + np.random.normal(0, 0.002)) * 0.24),
                    'Cost_EUR': max(0.0001, base_cost + np.random.normal(0, 0.001))
                })
        
        df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['Model_Size'] = df['Model_Name'].apply(lambda x: 
        '8B' if '8B' in x else '20B' if '20B' in x or 'Small' in x else '70B+')
    df['Quality_per_EUR'] = df['Quality_Score'] / df['Cost_EUR']
    df['Quality_per_kWh'] = df['Quality_Score'] / df['Energy_kWh']
    df['Efficiency_Score'] = (df['Quality_Score'] / 5) * 0.4 + \
                             (1 - df['Cost_EUR'] / df['Cost_EUR'].max()) * 0.2 + \
                             (1 - df['Energy_kWh'] / df['Energy_kWh'].max()) * 0.2 + \
                             (1 - df['Latency_sec'] / df['Latency_sec'].max()) * 0.2
    
    return df

# Load data - automatically connects to Excel file
df = load_data()

# Optional: Add manual refresh button in sidebar
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.rerun()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["Overview", "Charts", "Comparison Table", "Recommendations"]
)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
selected_models = st.sidebar.multiselect(
    "Models:",
    options=df['Model_Name'].unique(),
    default=df['Model_Name'].unique(),
    help="Select which AI models to include in the analysis"
)
selected_categories = st.sidebar.multiselect(
    "Task Categories:",
    options=df['Category'].unique(),
    default=df['Category'].unique(),
    help="Select which types of tasks to analyze"
)

# Apply filters
filtered_df = df[
    (df['Model_Name'].isin(selected_models)) & 
    (df['Category'].isin(selected_categories))
]

# Check if filtered data is empty
if filtered_df.empty:
    st.error("No data matches the selected filters. Please adjust your filters.")
    st.stop()

# Add explanatory section
st.sidebar.markdown("---")
st.sidebar.subheader("Key Terms")
with st.sidebar.expander("Click to learn about AI metrics"):
    st.markdown("""
    **Quality Score**: How well the AI performs the task (1-5 scale)
    
    **Latency**: Response time - how fast the AI gives an answer (in seconds)
    
    **Energy**: Power consumption needed to run the AI (kWh = kilowatt-hours)
    
    **Cost**: Estimated operational cost in euros
    
    **CO₂**: Carbon dioxide emissions from energy use (environmental impact)
    
    **Model Size**: 
    - 8B = Small, fast models
    - 20B = Medium-sized models  
    - 70B+ = Large, powerful models
    """)

# PAGE 1: OVERVIEW
if page == "Overview":
    st.title("AI Model Comparison Dashboard")
    st.markdown("### Comprehensive analysis of AI language models across 30 different tasks")
    
    # Add explanation
    st.info("""
    This dashboard compares different AI language models (like ChatGPT, Claude, etc.) to help you understand:
    - Which models perform best for different types of tasks
    - How much they cost to run and their environmental impact
    - Trade-offs between performance, speed, and efficiency
    """)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_quality = filtered_df['Quality_Score'].mean()
        st.metric(
            label="Average Quality",
            value=f"{avg_quality:.2f}/5",
            delta=f"{(avg_quality - 3.5):.2f} vs baseline",
            help="How well the AI models perform on average (higher is better)"
        )
    
    with col2:
        total_cost = filtered_df['Cost_EUR'].sum()
        st.metric(
            label="Total Cost",
            value=f"{total_cost:.3f} €",
            delta=f"-{((1 - total_cost/10)*100):.0f}% vs budget",
            help="Estimated cost to run all selected models on all tasks"
        )
    
    with col3:
        total_energy = filtered_df['Energy_kWh'].sum()
        st.metric(
            label="Total Energy",
            value=f"{total_energy:.3f} kWh",
            delta=f"{total_energy*0.24:.2f} kg CO₂",
            help="Energy consumption and resulting carbon emissions"
        )
    
    with col4:
        avg_latency = filtered_df['Latency_sec'].mean()
        st.metric(
            label="Average Response Time",
            value=f"{avg_latency:.2f} sec",
            delta="Real-time" if avg_latency < 1 else "Acceptable",
            help="How fast the AI models respond (lower is better)"
        )
    
    st.markdown("---")
    
    # Summary by model
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance by Model")
        st.caption("Average performance metrics for each AI model")
        model_summary = filtered_df.groupby('Model_Name').agg({
            'Quality_Score': 'mean',
            'Latency_sec': 'mean',
            'Energy_kWh': 'sum',
            'Cost_EUR': 'sum'
        }).round(3)
        model_summary.columns = ['Avg Quality', 'Avg Response Time (s)', 'Total Energy (kWh)', 'Total Cost (€)']
        st.dataframe(model_summary, use_container_width=True)
    
    with col2:
        st.subheader("Performance by Task Category")
        st.caption("How well models perform on different types of tasks")
        category_summary = filtered_df.groupby('Category').agg({
            'Quality_Score': 'mean',
            'Latency_sec': 'mean',
            'Energy_kWh': 'mean'
        }).round(3)
        category_summary.columns = ['Avg Quality', 'Avg Response Time (s)', 'Avg Energy (kWh)']
        st.dataframe(category_summary, use_container_width=True)
    
    # Radar chart comparison
    st.markdown("---")
    st.subheader("Multi-dimensional Comparison")
    st.caption("Compare models across multiple performance dimensions (closer to edge = better)")
    
    radar_data = filtered_df.groupby('Model_Name').agg({
        'Quality_Score': 'mean',
        'Efficiency_Score': 'mean'
    }).reset_index()
    
    radar_data['Speed_Score'] = 1 - (filtered_df.groupby('Model_Name')['Latency_sec'].mean() / 
                                      filtered_df.groupby('Model_Name')['Latency_sec'].mean().max()).values
    radar_data['Cost_Score'] = 1 - (filtered_df.groupby('Model_Name')['Cost_EUR'].sum() / 
                                     filtered_df.groupby('Model_Name')['Cost_EUR'].sum().max()).values
    radar_data['Energy_Score'] = 1 - (filtered_df.groupby('Model_Name')['Energy_kWh'].sum() / 
                                       filtered_df.groupby('Model_Name')['Energy_kWh'].sum().max()).values
    
    fig_radar = go.Figure()
    
    for model in radar_data['Model_Name']:
        model_data = radar_data[radar_data['Model_Name'] == model]
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                model_data['Quality_Score'].values[0] / 5,
                model_data['Speed_Score'].values[0],
                model_data['Cost_Score'].values[0],
                model_data['Energy_Score'].values[0],
                model_data['Efficiency_Score'].values[0]
            ],
            theta=['Quality', 'Speed', 'Cost', 'Energy', 'Efficiency'],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# PAGE 2: CHARTS
elif page == "Charts":
    st.title("Detailed Chart Analysis")
    st.caption("Explore relationships between different performance metrics")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Quality vs Energy", 
        "Quality vs Cost", 
        "Response Time vs Model Size",
        "Distributions"
    ])
    
    with tab1:
        st.subheader("Quality vs Energy Consumption")
        st.caption("Shows the trade-off between AI performance and environmental impact")
        fig1 = px.scatter(
            filtered_df,
            x='Energy_kWh',
            y='Quality_Score',
            color='Model_Name',
            size='Cost_EUR',
            hover_data=['Task_Description', 'Latency_sec'],
            title="Quality-Energy Trade-off",
            labels={
                'Energy_kWh': 'Energy Consumption (kWh)',
                'Quality_Score': 'Quality Score (1-5)',
                'Cost_EUR': 'Cost (€)'
            }
        )
        fig1.update_layout(height=600)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.info("💡 **Interpretation**: Models in the top-left are ideal (high quality, low energy use)")
        
        # Export button
        st.download_button(
            label="Download Data",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='quality_energy_data.csv',
            mime='text/csv'
        )
    
    with tab2:
        st.subheader("Quality vs Operational Cost")
        st.caption("Find models that give the best value for money")
        fig2 = px.scatter(
            filtered_df,
            x='Cost_EUR',
            y='Quality_Score',
            color='Model_Size',
            size='Energy_kWh',
            hover_data=['Model_Name', 'Task_Description'],
            title="Quality-Cost Trade-off",
            labels={
                'Cost_EUR': 'Cost (€)',
                'Quality_Score': 'Quality Score (1-5)',
                'Model_Size': 'Model Size'
            },
            color_discrete_map={'8B': '#10b981', '20B': '#f59e0b', '70B+': '#ef4444'}
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.info("💡 **Interpretation**: Models in the top-left offer the best quality per euro spent")
    
    with tab3:
        st.subheader("Response Time by Model Size")
        st.caption("Larger models are typically slower but more capable")
        fig3 = px.box(
            filtered_df,
            x='Model_Size',
            y='Latency_sec',
            color='Model_Size',
            points="all",
            title="Response Time Distribution",
            labels={
                'Model_Size': 'Model Size',
                'Latency_sec': 'Response Time (seconds)'
            },
            color_discrete_map={'8B': '#10b981', '20B': '#f59e0b', '70B+': '#ef4444'}
        )
        fig3.update_layout(height=600)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.info("💡 **Model Size Guide**: 8B = Fast & efficient, 20B = Balanced, 70B+ = Powerful but slower")
        
        # Comparison line chart
        st.subheader("Response Time Trend Across Tasks")
        st.caption("How response time varies across different task types")
        latency_by_task = filtered_df.groupby(['Task_ID', 'Model_Name'])['Latency_sec'].mean().reset_index()
        fig3b = px.line(
            latency_by_task,
            x='Task_ID',
            y='Latency_sec',
            color='Model_Name',
            title="Response Time by Task",
            labels={
                'Task_ID': 'Task ID',
                'Latency_sec': 'Response Time (sec)',
                'Model_Name': 'Model'
            }
        )
        st.plotly_chart(fig3b, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Metric Distributions")
        st.caption("Understand the range and spread of different performance measures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4a = px.histogram(
                filtered_df,
                x='Quality_Score',
                color='Model_Name',
                nbins=20,
                title="Quality Score Distribution",
                labels={'Quality_Score': 'Quality Score'}
            )
            st.plotly_chart(fig4a, use_container_width=True)
            
            fig4c = px.violin(
                filtered_df,
                y='Energy_kWh',
                x='Model_Name',
                box=True,
                title="Energy Consumption Distribution",
                labels={'Energy_kWh': 'Energy (kWh)'}
            )
            st.plotly_chart(fig4c, use_container_width=True)
        
        with col2:
            fig4b = px.histogram(
                filtered_df,
                x='Cost_EUR',
                color='Model_Size',
                nbins=20,
                title="Cost Distribution",
                labels={'Cost_EUR': 'Cost (€)'},
                color_discrete_map={'8B': '#10b981', '20B': '#f59e0b', '70B+': '#ef4444'}
            )
            st.plotly_chart(fig4b, use_container_width=True)
            
            fig4d = px.box(
                filtered_df,
                x='Category',
                y='Quality_Score',
                color='Category',
                title="Quality by Task Category",
                labels={'Quality_Score': 'Quality Score'}
            )
            fig4d.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig4d, use_container_width=True)

# PAGE 3: COMPARISON TABLE
elif page == "Comparison Table":
    st.title("Rankings and Detailed Comparison")
    st.caption("Comprehensive model rankings and head-to-head comparisons")
    
    # Overall ranking
    st.subheader("Overall Model Rankings")
    st.caption("Models ranked by overall efficiency score (combines quality, cost, energy, and speed)")
    
    ranking = filtered_df.groupby('Model_Name').agg({
        'Quality_Score': 'mean',
        'Latency_sec': 'mean',
        'Energy_kWh': 'sum',
        'CO2_kg': 'sum',
        'Cost_EUR': 'sum',
        'Efficiency_Score': 'mean',
        'Quality_per_EUR': 'mean',
        'Quality_per_kWh': 'mean'
    }).round(3)
    
    ranking = ranking.sort_values('Efficiency_Score', ascending=False)
    ranking['Rank'] = range(1, len(ranking) + 1)
    ranking = ranking[['Rank', 'Quality_Score', 'Latency_sec', 'Energy_kWh', 
                       'CO2_kg', 'Cost_EUR', 'Efficiency_Score', 
                       'Quality_per_EUR', 'Quality_per_kWh']]
    
    ranking.columns = ['Rank', 'Avg Quality', 'Avg Response Time (s)', 'Total Energy (kWh)', 
                       'Total CO₂ (kg)', 'Total Cost (€)', 'Efficiency Score', 
                       'Quality per €', 'Quality per kWh']
    
    # Color code the ranking
    def color_ranking(val):
        if isinstance(val, (int, float)):
            if val <= 2:
                return 'background-color: #10b981; color: white'
            elif val <= 4:
                return 'background-color: #f59e0b; color: white'
            else:
                return 'background-color: #ef4444; color: white'
        return ''
    
    styled_ranking = ranking.style.map(color_ranking, subset=['Rank'])
    st.dataframe(styled_ranking, use_container_width=True)
    
    # Add ranking explanation
    st.info("""
    **Ranking Guide**: 
    - 🟢 **Green (1-2)**: Top performers - Best overall value
    - 🟡 **Yellow (3-4)**: Good performers - Solid choice for most use cases  
    - 🔴 **Red (5+)**: Specialized use only - May excel in specific areas but lower overall efficiency
    """)
    
    # Download button
    st.download_button(
        label="Download Rankings (CSV)",
        data=ranking.to_csv().encode('utf-8'),
        file_name='ranking_comparia.csv',
        mime='text/csv'
    )
    
    st.markdown("---")
    
    # Detailed comparison table
    st.subheader("Detailed Task-by-Task Comparison")
    st.caption("Compare how different models perform on specific tasks")
    
    selected_tasks = st.multiselect(
        "Select tasks to compare:",
        options=filtered_df['Task_ID'].unique(),
        default=filtered_df['Task_ID'].unique()[:5],
        help="Choose specific tasks to see detailed model comparisons"
    )
    
    if selected_tasks:
        comparison_df = filtered_df[filtered_df['Task_ID'].isin(selected_tasks)][
            ['Task_ID', 'Task_Description', 'Category', 'Model_Name', 
             'Quality_Score', 'Latency_sec', 'Energy_kWh', 'Cost_EUR']
        ]
        
        # Pivot table
        pivot_quality = comparison_df.pivot_table(
            index=['Task_ID', 'Task_Description'],
            columns='Model_Name',
            values='Quality_Score',
            aggfunc='mean'
        ).round(2)
        
        st.dataframe(pivot_quality, use_container_width=True)
        
        # Heatmap
        st.caption("Darker green = better performance for that model on that task")
        fig_heatmap = px.imshow(
            pivot_quality.values,
            x=pivot_quality.columns,
            y=[f"T{idx}" for idx in pivot_quality.index.get_level_values(0)],
            color_continuous_scale='RdYlGn',
            labels=dict(x="Model", y="Task", color="Quality"),
            title="Quality Heatmap by Task and Model"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # Category performance
    st.subheader("Performance by Task Category")
    st.caption("How different models perform across various types of AI tasks")
    
    category_performance = filtered_df.groupby(['Category', 'Model_Name']).agg({
        'Quality_Score': 'mean',
        'Energy_kWh': 'mean',
        'Cost_EUR': 'mean'
    }).reset_index()
    
    fig_category = px.bar(
        category_performance,
        x='Category',
        y='Quality_Score',
        color='Model_Name',
        barmode='group',
        title="Quality Score by Task Category",
        labels={'Quality_Score': 'Average Quality', 'Category': 'Task Category'}
    )
    fig_category.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_category, use_container_width=True)

# PAGE 4: RECOMMENDATIONS
else:
    st.title("Recommendations & Insights")
    st.caption("AI-powered recommendations to help you choose the right model for your needs")
    
    # Check if we have enough data for recommendations
    if len(filtered_df['Model_Name'].unique()) < 2:
        st.warning("Select at least 2 different models to get comparative recommendations.")
        st.stop()
    
    # Add explanatory section
    st.info("""
    **How to interpret recommendations**: We analyze performance across quality, cost, energy efficiency, and speed 
    to identify the best models for different use cases. Consider your priorities when making decisions.
    """)
    
    # Calculate best models for different scenarios
    best_quality = filtered_df.groupby('Model_Name')['Quality_Score'].mean().idxmax()
    best_cost = filtered_df.groupby('Model_Name')['Cost_EUR'].sum().idxmin()
    best_energy = filtered_df.groupby('Model_Name')['Energy_kWh'].sum().idxmin()
    best_efficiency = filtered_df.groupby('Model_Name')['Efficiency_Score'].mean().idxmax()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Highest Quality</h3>
            <h2>{best_quality}</h2>
            <p>Score: {filtered_df[filtered_df['Model_Name']==best_quality]['Quality_Score'].mean():.2f}/5</p>
            <small>Best for tasks requiring maximum accuracy</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Most Cost-Effective</h3>
            <h2>{best_cost}</h2>
            <p>Total cost: {filtered_df[filtered_df['Model_Name']==best_cost]['Cost_EUR'].sum():.4f} €</p>
            <small>Best for budget-conscious applications</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Most Eco-Friendly</h3>
            <h2>{best_energy}</h2>
            <p>Energy: {filtered_df[filtered_df['Model_Name']==best_energy]['Energy_kWh'].sum():.4f} kWh</p>
            <small>Best for environmentally conscious deployment</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Overall Balance</h3>
            <h2>{best_efficiency}</h2>
            <p>Score: {filtered_df[filtered_df['Model_Name']==best_efficiency]['Efficiency_Score'].mean():.3f}</p>
            <small>Best compromise across all metrics</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Use case recommendations
    st.subheader("Recommendations by Use Case")
    st.caption("Choose the right model size based on your specific needs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Simple Tasks")
        st.markdown("""
        **Recommended:** Small models (8B parameters)
        
        **Best for:**
        - Basic Q&A and factual queries
        - Simple text translation
        - Content classification
        - Information extraction
        - Quick responses needed
        
        **Advantages:**
        - Very fast response times (<1s)
        - Minimal cost and energy use
        - Good for high-volume applications
        """)
    
    with col2:
        st.markdown("### Moderate Complexity")
        st.markdown("""
        **Recommended:** Medium models (20B parameters)
        
        **Best for:**
        - Problem-solving tasks
        - Code generation (simple)
        - Text analysis and summarization
        - Complex reasoning
        - Creative writing assistance
        
        **Advantages:**
        - Good quality-to-cost ratio
        - Acceptable response times
        - Stable performance across tasks
        """)
    
    with col3:
        st.markdown("### Complex Tasks")
        st.markdown("""
        **Recommended:** Large models (70B+ parameters)
        
        **Best for:**
        - Advanced reasoning and analysis
        - Creative content generation
        - Multi-step problem solving
        - Research and deep analysis
        - Mission-critical applications
        
        **Trade-offs:**
        - Highest quality output
        - Higher cost and energy use
        - Slower response times
        """)
        
    
    st.markdown("---")
    
    # Key insights
    st.subheader("Key Insights")
    st.caption("Data-driven insights from the analysis")
    
    insights = []
    
    # Quality variance
    quality_std = filtered_df.groupby('Model_Name')['Quality_Score'].std()
    most_consistent = quality_std.idxmin()
    insights.append(f"**Most consistent model:** {most_consistent} (standard deviation: {quality_std.min():.2f})")
    
    # Energy efficiency
    best_quality_energy = filtered_df.groupby('Model_Name')['Quality_per_kWh'].mean().idxmax()
    insights.append(f"**Best quality-to-energy ratio:** {best_quality_energy}")
    
    # Cost efficiency
    best_quality_cost = filtered_df.groupby('Model_Name')['Quality_per_EUR'].mean().idxmax()
    insights.append(f"**Best quality-to-price ratio:** {best_quality_cost}")
    
    # Category performance
    hardest_category = filtered_df.groupby('Category')['Quality_Score'].mean().idxmin()
    insights.append(f"**Most challenging task category:** {hardest_category}")
    
    easiest_category = filtered_df.groupby('Category')['Quality_Score'].mean().idxmax()
    insights.append(f"**Easiest task category:** {easiest_category}")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.markdown("---")
    
    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    st.caption("Best practices for implementing AI models in production")
    
    st.markdown("""
    ### Optimal Deployment Strategy
    
    1. **Hybrid Approach Recommended:**
       - Use small models (8B) to filter and classify incoming requests
       - Automatically route to appropriate model based on task complexity
       - Reserve large models for critical tasks only
    
    2. **Cost Optimization:**
       - Implement caching for frequent queries
       - Use batch processing for non-urgent tasks
       - Continuous monitoring to identify overuse
    
    3. **Green AI Best Practices:**
       - Prefer data centers with low carbon intensity
       - Schedule heavy tasks during off-peak hours
       - Quantify and offset residual emissions
    
    4. **Selection Criteria:**
       - **Latency-critical:** Use 8B-20B models
       - **Budget-limited:** Use 8B models with fine-tuning
       - **Maximum quality:** Use 70B+ models for specific use cases
       - **Optimal balance:** Use 20B models for general purposes
    """)
    
    # Export full report
    st.markdown("---")
    st.subheader("Complete Export")
    st.caption("Download comprehensive analysis data for further processing")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Full Report (CSV)",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='ai_models_full_report.csv',
            mime='text/csv',
            help="Complete dataset with all metrics and calculations"
        )
    
    with col2:
        ranking_export = filtered_df.groupby('Model_Name').agg({
            'Quality_Score': 'mean',
            'Latency_sec': 'mean',
            'Energy_kWh': 'sum',
            'CO2_kg': 'sum',
            'Cost_EUR': 'sum',
            'Efficiency_Score': 'mean'
        }).round(3)
        st.download_button(
            label="Download Rankings (CSV)",
            data=ranking_export.to_csv().encode('utf-8'),
            file_name='ai_models_rankings.csv',
            mime='text/csv',
            help="Summary rankings and performance metrics"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px;'>
    <p><strong>AI Model Comparison Dashboard</strong> - Comprehensive LLM Analysis Platform</p>
    <p>Benchmark conducted across 30 diverse tasks | Green AI Course 2025</p>
    <p>Real-time data updates | Focus on energy efficiency and sustainability</p>
</div>
""", unsafe_allow_html=True)