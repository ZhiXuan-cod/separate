# components/account.py
import streamlit as st

from utils.auth import hash_password, verify_password


def account_page() -> None:
    """User account settings — view profile and change password."""
    st.markdown('<h2 class="sub-header">👤 Account Settings</h2>', unsafe_allow_html=True)

    st.markdown("### Your Profile")
    st.write(f"**Name:** {st.session_state.user_name}")
    st.write(f"**Email:** {st.session_state.user_email}")

    st.markdown("---")
    st.markdown("### Change Password")

    with st.form("change_password_form"):
        current_pw = st.text_input("Current password", type="password")
        new_pw = st.text_input("New password", type="password")
        confirm_pw = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update Password", use_container_width=True)

    if submitted:
        if not current_pw or not new_pw or not confirm_pw:
            st.error("Please fill in all three fields.")
        elif new_pw != confirm_pw:
            st.error("New passwords do not match.")
        elif len(new_pw) < 6:
            st.error("Password must be at least 6 characters.")
        elif st.session_state.get("supabase") is None:
            st.error("Database not connected — cannot update password.")
        else:
            try:
                resp = (
                    st.session_state.supabase
                    .table("users")
                    .select("*")
                    .eq("email", st.session_state.user_email)
                    .execute()
                )
                if not resp.data:
                    st.error("User record not found.")
                else:
                    user = resp.data[0]
                    if verify_password(current_pw, user.get("password", "")):
                        (
                            st.session_state.supabase
                            .table("users")
                            .update({"password": hash_password(new_pw)})
                            .eq("email", st.session_state.user_email)
                            .execute()
                        )
                        st.success("✅ Password updated successfully!")
                    else:
                        st.error("Current password is incorrect.")
            except Exception as exc:
                st.error(f"Failed to update password: {exc}")
