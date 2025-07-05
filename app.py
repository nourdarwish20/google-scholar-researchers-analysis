import glob
import html
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures

APP_TITLE = "LU Researchers Explorer"
PROFILES_CSV = "profiles.csv"
CORPUS_DIR = "researcher_corpora"
UNCLASSIFIED = "Unclassified"
SCHOLAR_ID_PATTERN = r'user=([\w-]+)'
FORECAST_FIT_WINDOW = 15   # years of history used to fit the trend
MAX_ARTICLES_SHOWN = 200   # keep the page responsive for very prolific authors


@st.cache_data
def load_profiles():
    try:
        profiles = pd.read_csv(PROFILES_CSV)
    except Exception as e:
        st.error(f"Failed to load {PROFILES_CSV}: {e}")
        return pd.DataFrame()
    profiles['Scholar ID'] = profiles['Profile URL'].str.extract(SCHOLAR_ID_PATTERN)[0]
    # The scraper writes "Not found" for missing metrics
    for col in ['Total Citations', 'H-index (Total)', 'Total Papers']:
        profiles[col] = pd.to_numeric(profiles[col], errors='coerce')
    return profiles


@st.cache_data
def load_articles():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.csv")))
    if not files:
        st.error(f"No publication files found in {CORPUS_DIR}/")
        return pd.DataFrame()
    df_list = []
    for file in files:
        try:
            df_list.append(pd.read_csv(file))
        except Exception as e:
            st.warning(f"Error reading {file}: {e}")
    articles = pd.concat(df_list, ignore_index=True)
    articles['Scholar ID'] = articles['Scholar Link'].str.extract(SCHOLAR_ID_PATTERN)[0]
    return articles


@st.cache_data
def build_researchers(profiles_df, articles_df):
    """One row per researcher, joined to profile metrics by Google Scholar user ID.

    Joining on the ID instead of the name avoids mismatches caused by
    punctuation, titles ("Ph.D.") or profiles whose name failed to scrape.
    """
    researchers = (
        articles_df.groupby('Researcher Name', as_index=False)['Scholar ID'].first()
    )
    metrics = profiles_df[['Scholar ID', 'Profile URL', 'Total Citations',
                           'H-index (Total)', 'Total Papers']]
    researchers = researchers.merge(metrics, on='Scholar ID', how='left')
    fallback_url = "https://scholar.google.com/citations?hl=en&user=" + researchers['Scholar ID']
    researchers['Profile URL'] = researchers['Profile URL'].fillna(fallback_url)
    return researchers.sort_values('Researcher Name').reset_index(drop=True)


def name_matches(name, query):
    """Every word of the query must be the start of a word in the name.

    "ali" matches "Ali Awdeh" but not "Khalil"; "haj hass" matches
    "Fouad El Haj Hassan". Plain string comparison, so punctuation in
    names or queries is never interpreted as a regex.
    """
    name_words = str(name).lower().replace('-', ' ').split()
    return all(any(w.startswith(q) for w in name_words) for q in query.lower().split())


@st.cache_data
def compute_researcher_similarity(articles_df):
    """Cosine similarity between researchers based on their classified topics.

    Unclassified / missing labels are excluded, so researchers are never
    considered similar just because many of their papers were not classified.
    """
    classified = articles_df[
        articles_df['Classified_Topic'].notna()
        & (articles_df['Classified_Topic'] != UNCLASSIFIED)
    ]
    if classified.empty:
        return None

    topic_matrix = pd.crosstab(classified['Researcher Name'], classified['Classified_Topic'])
    return pd.DataFrame(
        cosine_similarity(topic_matrix),
        index=topic_matrix.index,
        columns=topic_matrix.index,
    )


def top_topics(articles_df, researcher_name, n=3):
    topics = articles_df.loc[
        (articles_df['Researcher Name'] == researcher_name)
        & (articles_df['Classified_Topic'] != UNCLASSIFIED),
        'Classified_Topic',
    ].value_counts().head(n)
    return ", ".join(topics.index) if not topics.empty else "No classified topics"


def fmt_metric(value):
    return "N/A" if pd.isna(value) else f"{int(value):,}"


def display_similar_researchers(researcher_name, similarity_matrix, researchers_df, articles_df, top_n=5):
    scores = similarity_matrix[researcher_name].drop(researcher_name)
    top_scores = scores[scores > 0].sort_values(ascending=False).head(top_n)

    if top_scores.empty:
        st.info("No researchers share classified topics with this researcher.")
        return

    st.subheader(f"Researchers Similar to {researcher_name}")
    st.caption("Similarity is the cosine similarity of the researchers' classified publication topics "
               "(unclassified articles are ignored).")

    for name, score in top_scores.items():
        profile = researchers_df[researchers_df['Researcher Name'] == name].iloc[0]

        with st.container():
            st.markdown(f"#### [{name}]({profile['Profile URL']})")
            st.markdown(f"**Similarity:** `{score:.3f}`")

            cols = st.columns(3)
            cols[0].metric("H-Index", fmt_metric(profile['H-index (Total)']))
            cols[1].metric("Citations", fmt_metric(profile['Total Citations']))
            cols[2].metric("Papers", fmt_metric(profile['Total Papers']))
            st.markdown(f"**Top topics:** {top_topics(articles_df, name)}")

            st.progress(float(min(score, 1.0)))
            st.markdown("---")


def display_profiles(researchers_df):
    if researchers_df.empty:
        st.info("No matching author profile found.")
        return
    for _, row in researchers_df.iterrows():
        name = html.escape(row['Researcher Name'])
        url = html.escape(row['Profile URL'], quote=True)
        st.markdown(f'### <a href="{url}" target="_blank" class="author-link">{name}</a>',
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex; gap: 20px; flex-wrap: wrap;'>"
            f"<span><b>Total Citations:</b> {fmt_metric(row['Total Citations'])}</span>"
            f"<span><b>Total Papers:</b> {fmt_metric(row['Total Papers'])}</span>"
            f"<span><b>H-Index:</b> {fmt_metric(row['H-index (Total)'])}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def display_articles(articles_df):
    if len(articles_df) > MAX_ARTICLES_SHOWN:
        st.caption(f"Showing the first {MAX_ARTICLES_SHOWN} of {len(articles_df)} articles. "
                   "Use the topic filter to narrow the list.")

    for _, row in articles_df.head(MAX_ARTICLES_SHOWN).iterrows():
        title = row.get('Title') if pd.notna(row.get('Title')) else 'No Title'
        topic = html.escape(str(row.get('Classified_Topic', UNCLASSIFIED)))
        link = html.escape(str(row.get('Scholar Link', '#')), quote=True)
        description = row.get('Description') if pd.notna(row.get('Description')) else 'No description'
        co_authors = row.get('Authors') if pd.notna(row.get('Authors')) else 'N/A'
        citations = row.get('Total citations') if pd.notna(row.get('Total citations')) else 'N/A'

        with st.expander(f"{title} — {row['Researcher Name']}", expanded=False):
            st.markdown(f"**Article Link:** <a href='{link}' target='_blank' style='color:#276678;'>click here</a>",
                        unsafe_allow_html=True)
            st.markdown(f"**Description:** {description}")
            st.markdown(f"**Classified Topic:** <code>{topic}</code>", unsafe_allow_html=True)
            st.markdown(f"**Co-authors:** {co_authors}")
            st.markdown(f"**Cited by:** {citations}")


def publication_years(articles_df):
    # Dates look like "2012" or "2012/3/25"; the year is always the first 4 digits
    return articles_df['Publication date'].astype(str).str.extract(r'^(\d{4})')[0].dropna().astype(int)


def publication_forecast(articles_df, forecast_years=5, exclude_year=None):
    """Fit a degree-2 polynomial to yearly publication counts and extrapolate.

    Years with no publications count as 0 (they are not simply skipped), and
    only the last FORECAST_FIT_WINDOW years are used so a single very old
    paper does not distort the trend. Publications from `exclude_year` onwards
    (the incomplete snapshot year) are ignored.
    """
    years = publication_years(articles_df)
    if exclude_year is not None:
        years = years[years < exclude_year]
    if years.empty:
        return None

    last_year = years.max()
    first_year = max(years.min(), last_year - FORECAST_FIT_WINDOW + 1)

    pub_counts = years.value_counts().reindex(range(first_year, last_year + 1), fill_value=0)
    if (pub_counts > 0).sum() < 3:
        return None

    X = pub_counts.index.values.reshape(-1, 1)
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1).reshape(-1, 1)
    # Fit on years relative to first_year: raw years (~2000^2) make year and year^2
    # almost collinear, and the ill-conditioned fit gives version-dependent results
    poly = PolynomialFeatures(degree=2)
    model = LinearRegression().fit(poly.fit_transform(X - first_year), pub_counts.values)
    forecast = np.clip(model.predict(poly.transform(future_years - first_year)), 0, None)

    history_df = pd.DataFrame({'Year': X.flatten(), 'Publications': pub_counts.values, 'Type': 'Historical'})
    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Publications': forecast, 'Type': 'Forecast'})
    return pd.concat([history_df, forecast_df], ignore_index=True)


def forecast_chart(forecast_data):
    """Solid line for history, dashed line for the forecast, and a rule at the boundary."""
    history = forecast_data[forecast_data['Type'] == 'Historical']
    # Start the forecast line at the last historical point so the two lines connect
    forecast = pd.concat([history.tail(1).assign(Type='Forecast'),
                          forecast_data[forecast_data['Type'] == 'Forecast']])
    plot_df = pd.concat([history, forecast])

    color = alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'],
                                                range=['#073b4c', '#e07a5f']),
                      legend=alt.Legend(title=None, orient='top'))
    base = alt.Chart(plot_df).encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('Publications:Q', title='Publications per year', scale=alt.Scale(zero=True)),
        color=color,
        tooltip=['Year', 'Type', alt.Tooltip('Publications:Q', format='.1f')],
    )
    lines = base.mark_line(point=True).encode(
        strokeDash=alt.StrokeDash('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'],
                                                            range=[[1, 0], [6, 4]]), legend=None)
    )
    boundary = alt.Chart(pd.DataFrame({'Year': [history['Year'].max()]})).mark_rule(
        color='gray', strokeDash=[2, 2]
    ).encode(x='Year:O')
    return (lines + boundary).properties(height=350)


def set_background():
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #12283a;
    }
    .author-link, a {
        color: #1b3559;
        text-decoration: none;
        font-weight: 700;
        transition: color 0.3s ease;
    }
    .author-link:hover, a:hover {
        color: #0a2540;
        text-decoration: underline;
    }
    code {
        background-color: #7da9c0;
        padding: 3px 7px;
        border-radius: 5px;
        color: #073b4c;
        font-weight: 600;
    }
    .stSidebar {
        background-color: #d4e2f4;
        color: #12283a;
        font-weight: 600;
    }
    .section-header {
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid #1b3559;
    }
    </style>
    """, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f"<h2 class='section-header'>{text}</h2>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    set_background()
    st.title(APP_TITLE)

    profiles_df = load_profiles()
    articles_df = load_articles()
    if articles_df.empty or profiles_df.empty:
        st.stop()
    researchers_df = build_researchers(profiles_df, articles_df)

    with st.spinner("Analyzing researcher similarities..."):
        similarity_matrix = compute_researcher_similarity(articles_df)

    st.sidebar.header("Search & Filter")
    author_input = st.sidebar.text_input("Author Name (for Profile & Articles)")
    classifications = sorted(articles_df['Classified_Topic'].dropna().unique())
    selected_classification = st.sidebar.selectbox("Filter by Classification", options=["All"] + classifications)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Researcher Similarity")
    similarity_researcher = None
    if similarity_matrix is not None:
        similarity_researcher = st.sidebar.selectbox(
            "Find researchers similar to:",
            options=["Select a researcher"] + sorted(similarity_matrix.index),
        )
        top_n = st.sidebar.slider("Number of similar researchers", 1, 10, 5)
    else:
        st.sidebar.info("Topic data not available for similarity analysis")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Forecasting Options")
    forecast_years = st.sidebar.slider("Years to forecast", 3, 10, 5)
    show_forecast = st.sidebar.checkbox("Show publication forecast", True)
    exclude_last_year = st.sidebar.checkbox(
        "Exclude the snapshot's latest year (incomplete)", True,
        help="The dataset is a snapshot taken part-way through its latest year.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This application allows you to explore researchers from the Lebanese University. "
        "You can search for an author to view their profile and published articles, "
        "filter articles by topic, find similar researchers, and see publication forecasts."
    )

    # Researcher similarity view replaces the regular view when a researcher is selected
    if similarity_researcher and similarity_researcher != "Select a researcher":
        section_header("Researcher Similarity Analysis")
        st.subheader(f"Selected Researcher: {similarity_researcher}")
        display_profiles(researchers_df[researchers_df['Researcher Name'] == similarity_researcher])
        display_similar_researchers(similarity_researcher, similarity_matrix,
                                    researchers_df, articles_df, top_n)
        return

    author_query = author_input.strip()
    filtered_articles = articles_df

    if author_query:
        matched = researchers_df[researchers_df['Researcher Name'].apply(name_matches, query=author_query)]
        filtered_articles = filtered_articles[filtered_articles['Researcher Name'].isin(matched['Researcher Name'])]
        section_header("Author Profile")
        display_profiles(matched)

    if selected_classification != "All":
        filtered_articles = filtered_articles[filtered_articles['Classified_Topic'] == selected_classification]

    if not (author_query or selected_classification != "All"):
        st.info("Please use the sidebar controls to explore researchers and articles.")
        return

    section_header("Articles")
    if filtered_articles.empty:
        st.info("No articles found with current filters.")
        return
    display_articles(filtered_articles)

    if show_forecast:
        section_header("Publication Forecast")
        # The snapshot's latest year is incomplete for everyone, so it is taken from the full dataset
        snapshot_year = publication_years(articles_df).max()
        forecast_data = publication_forecast(filtered_articles, forecast_years,
                                             snapshot_year if exclude_last_year else None)
        if forecast_data is None:
            st.warning("Insufficient data for forecasting (need at least 3 years with publications).")
            return

        st.altair_chart(forecast_chart(forecast_data), width='stretch')
        st.caption(
            f"**Forecast methodology:** publications per year over the last {FORECAST_FIT_WINDOW} years "
            "(years with no publications count as 0) are fitted with a degree-2 polynomial regression, "
            "which is then extrapolated. Negative predictions are clipped to 0. This is a simple trend "
            "extrapolation for illustration, not a statistical prediction."
        )
        with st.expander("View forecast details"):
            forecast_table = forecast_data.copy()
            forecast_table['Publications'] = forecast_table['Publications'].round().astype(int)
            st.dataframe(forecast_table, hide_index=True)


if __name__ == "__main__":
    main()
